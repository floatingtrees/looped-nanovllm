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
from nanovllm.layers.row_ops import copy_rows
from nanovllm.utils.context import get_context, set_depth


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
        # One Attention for every loop step: which step's kv cache it reads and writes
        # is data, set per row by set_depth, not a choice of module.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
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
        # recurrence() reports this gate's logit after every loop step; nothing acts on
        # it yet, so every step always runs.
        self.early_exit_gate = ReplicatedLinear(config.hidden_size, 1, bias=True)


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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.prelude(input_ids)
        for ut_step in range(self.model.total_ut_steps):
            depth = torch.full_like(positions, ut_step)
            hidden_states, _ = self.recurrence(hidden_states, positions, depth)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # forward and compute_logits, split at the loop boundary; forward is built from these.

    def prelude(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def recurrence(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention is depth blind; set depth configures the KV cache to assign 
        # depth to each sample
        set_depth(depth)
        for layer in self.model.layers:
            hidden_states = layer(positions, hidden_states)
        # The norm's output is both this step's readout and the next step's input, so
        # it belongs to the step rather than to the coda.
        hidden_states = self.model.norm(hidden_states)
        return hidden_states, self.model.early_exit_gate(hidden_states)

    def coda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def prelude_into(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        input_ids: torch.Tensor,
        count: torch.Tensor,
    ):
        copy_rows(hidden_states, self.model.embed_tokens.weight, rows, input_ids, count, rows.size(0))

    def early_exit_protocol(
        self,
        rows: torch.Tensor,
        count: torch.Tensor,
        kv_slots: torch.Tensor,
        depth: torch.Tensor,
    ):
        get_context().kv_cache.fill_forward(rows, count, kv_slots, depth)
