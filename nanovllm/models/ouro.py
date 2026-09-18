import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoConfig, Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class OuroConfig(Qwen3Config):
    model_type = "ouro"

    def __init__(
        self,
        total_ut_steps: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.total_ut_steps = total_ut_steps


# Registering the config makes AutoConfig prefer this class over the checkpoint's
# remote code, so no trust_remote_code is needed to read config.json.
if "ouro" not in CONFIG_MAPPING:
    AutoConfig.register("ouro", OuroConfig)


class OuroSandwichNorm(RMSNorm):

    @torch.compile
    def forward_add(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight).add_(residual)


class OuroAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        total_ut_steps: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # A loop step re-attends with the same weights but different keys and values,
        # so every step gets its own kv cache, like a layer of the unrolled model.
        self.attn = nn.ModuleList([
            Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.num_kv_heads,
            )
            for _ in range(total_ut_steps)
        ])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ut_step: int,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn[ut_step](q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class OuroMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class OuroDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OuroConfig,
    ) -> None:
        super().__init__()
        self.self_attn = OuroAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            total_ut_steps=config.total_ut_steps,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = OuroMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroSandwichNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = OuroSandwichNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ut_step: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, ut_step)
        hidden_states = self.input_layernorm_2.forward_add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2.forward_add(hidden_states, residual)
        return hidden_states


class OuroModel(nn.Module):

    def __init__(
        self,
        config: OuroConfig,
    ) -> None:
        super().__init__()
        self.total_ut_steps = config.total_ut_steps
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OuroDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # forward runs every loop step and never evaluates this gate; recurrence()
        # reports its output per step without acting on it.
        self.early_exit_gate = ReplicatedLinear(config.hidden_size, 1, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for ut_step in range(self.total_ut_steps):
            for layer in self.layers:
                hidden_states = layer(positions, hidden_states, ut_step)
            hidden_states = self.norm(hidden_states)
        return hidden_states


class OuroForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: OuroConfig
    ) -> None:
        super().__init__()
        self.model = OuroModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        # The engine sizes the kv cache from num_hidden_layers and then hands one slot to
        # each module holding a kv cache, so it has to see the unrolled depth.
        if not getattr(config, "num_hidden_layers_unrolled", False):
            config.num_hidden_layers *= config.total_ut_steps
            config.num_hidden_layers_unrolled = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # forward and compute_logits, split at the loop boundary:
    # coda(recurrence(prelude(input_ids), positions)[0]) == compute_logits(forward(input_ids, positions))

    def prelude(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def recurrence(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the final hidden states and the exit gate's logit after every loop
        step, shaped (num_tokens, total_ut_steps). The gate only reads the hidden
        states, so the hidden states match forward exactly."""
        gate_logits = []
        for ut_step in range(self.model.total_ut_steps):
            for layer in self.model.layers:
                hidden_states = layer(positions, hidden_states, ut_step)
            # The final norm runs inside the loop: its output is both the step's readout
            # and the next step's input, so it belongs here rather than in the coda.
            hidden_states = self.model.norm(hidden_states)
            gate_logits.append(self.model.early_exit_gate(hidden_states))
        return hidden_states, torch.cat(gate_logits, dim=-1)

    def coda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
