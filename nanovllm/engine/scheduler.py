from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []
        self.preempted: list[Sequence] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            completes = remaining >= num_tokens
            if completes and len(self.running) >= self.max_num_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def postprocess_prefill(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        continuing = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            if self.append(seq, token_id):
                continuing.append(seq)
        return continuing

    def commit(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        continuing = []
        for seq, token_id in zip(seqs, token_ids):
            seq.num_scheduled_tokens = 1
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if self.append(seq, token_id):
                continuing.append(seq)
        return continuing

    def append(self, seq: Sequence, token_id: int) -> bool:
        seq.append_token(token_id)
        seq.is_prefill = False
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
            self.finished.append(seq)
            return False
        return True

    def allocate_slots(self, seqs: list[Sequence]) -> list[Sequence]:
        for seq in seqs:
            if seq.status != SequenceStatus.RUNNING:
                continue
            while not self.block_manager.can_append(seq):
                victim = next((other for other in reversed(self.running) if other is not seq), None)
                if victim is None:
                    self.preempt(seq)
                    break
                self.preempt(victim)
            else:
                self.block_manager.may_append(seq)
        return [seq for seq in seqs if seq.status == SequenceStatus.RUNNING]

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.preempted.append(seq)

    def take_released(self) -> tuple[list[Sequence], list[Sequence]]:
        finished, preempted = self.finished, self.preempted
        self.finished, self.preempted = [], []
        return finished, preempted
