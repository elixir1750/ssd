"""Standard synchronous DFlash using SSD's native Qwen3 target.

Stage 1 deliberately runs one sequence on one GPU in eager mode. It does not
retain MASK KV across rounds, build outcome trees, or perform async drafting.
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
import torch
from torch.nn import functional as F
from transformers import DynamicCache

from ssd.engine.block_manager import BlockManager
from ssd.engine.sequence import SequenceStatus
from ssd.engine.dflash_support import (
    accepted_features, draft_value, probabilities, sample_probs, verify_block,
)
from ssd.models.dflash import DFlashDraftModel


class DFlashRunner:
    def __init__(self, config, device=None, reserve_target=True):
        self.config = config
        self.device = torch.device("cuda:0" if device is None else device)
        dtype = config.hf_config.torch_dtype or torch.bfloat16
        self.model, info = DFlashDraftModel.from_pretrained(
            config.draft, torch_dtype=dtype, output_loading_info=True,
        )
        if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError(f"DFlash checkpoint did not load exactly: {info}")
        self.model.to(self.device).eval()
        self.target = None
        self.reset()
        # Hold space while the target sizes its paged KV pool. This is released
        # after target initialization for the growing draft KV and feature delta.
        dc = config.draft_hf_config
        length = config.max_model_len + config.speculate_k + 1
        kv_elements = 2 * dc.num_hidden_layers * dc.num_key_value_heads * dc.head_dim * length
        feature_elements = config.max_model_len * dc.hidden_size * len(config.dflash_target_layers)
        self._reservation = (torch.empty(kv_elements + feature_elements, device=self.device, dtype=dtype)
                             if reserve_target else None)

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    def bind_target(self, target):
        self.target = target
        self._reservation = None

    def reset(self):
        self.cache = DynamicCache()
        self.pending_features = None
        self.seq_id = None

    def prefill(self, seq, features):
        self.reset()
        self.seq_id = seq.seq_id
        self.pending_features = features.unsqueeze(0).clone()

    @torch.inference_mode()
    def propose(self, seq, k):
        if self.seq_id != seq.seq_id or self.pending_features is None:
            raise RuntimeError("Missing DFlash prefill or accepted target features")
        start = seq.num_tokens  # next position is the known recovery/anchor
        cached = self.cache.get_seq_length()
        if cached + self.pending_features.shape[1] != start:
            raise RuntimeError("DFlash feature delta is not contiguous with cached context")
        ids = torch.full((1, k + 1), self.model.mask_token_id, dtype=torch.long, device=self.device)
        ids[0, 0] = seq.recovery_token_id
        embeddings = F.embedding(ids, self.target.model.embed_tokens.weight)
        embeddings *= float(draft_value(self.model.config, "input_embedding_scale", 1.0))
        positions = torch.arange(cached, start + k + 1, device=self.device).unsqueeze(0)
        hidden = self.model(
            position_ids=positions, noise_embedding=embeddings,
            target_hidden=self.pending_features, past_key_values=self.cache, use_cache=True,
        )
        # Standard DFlash: retain only target-derived context, discard self KV.
        self.cache.crop(start)
        self.pending_features = None
        logits = self.model.compute_logits(
            hidden[:, 1:], lambda h: F.linear(h, self.target.lm_head.weight))
        temperature = seq.temperature if seq.draft_temperature is None else seq.draft_temperature
        q = probabilities(logits[0], temperature)
        return sample_probs(q), q

    def update(self, features, accepted):
        self.pending_features = accepted_features(features, accepted)


class DFlashScheduler:
    """Serial scheduler with only a target paged cache; draft owns DynamicCache.

    No prefix-cache hits: skipped target forwards would omit draft features.
    No preemption: it would otherwise change the original prompt accounting.
    """
    def __init__(self, config):
        self.config = config
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.waiting, self.running = deque(), deque()
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          max_model_len=config.max_model_len)

    def add(self, seq):
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def candidate_count(self, seq):
        return min(self.config.speculate_k,
                   seq.max_new_tokens - seq.num_completion_tokens - 1,
                   self.max_model_len - seq.num_tokens - 1)

    def schedule(self):
        if not self.running:
            seq = self.waiting.popleft()
            if not self.block_manager.can_allocate(seq):
                raise RuntimeError("Insufficient target KV space for this prompt")
            self.block_manager.hash_to_block_id.clear()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            return [seq], True
        seq = self.running[0]
        width = self.candidate_count(seq) + 1
        if not self.block_manager.can_append(seq, width):
            raise RuntimeError("Insufficient target KV space; lower max_model_len or model memory use")
        self.block_manager.may_append(seq, width)
        return [seq], False

    def commit(self, seq, suffix, recovery):
        remaining = min(seq.max_new_tokens - seq.num_completion_tokens,
                        self.max_model_len - seq.num_tokens)
        suffix = suffix[:remaining]
        if not seq.ignore_eos and self.eos in suffix:
            suffix = suffix[:suffix.index(self.eos) + 1]
        for token in suffix:
            seq.append_token(token)
        seq.num_cached_tokens = seq.num_tokens
        seq.recovery_token_id = recovery
        seq.last_spec_step_accepted_len = len(suffix)
        finished = ((not seq.ignore_eos and self.eos in suffix) or
                    seq.num_completion_tokens >= seq.max_new_tokens or
                    seq.num_tokens >= self.max_model_len)
        if finished:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
        return len(suffix)


class DFlashStep:
    def __init__(self, scheduler, target, draft, metrics):
        self.scheduler, self.target, self.draft, self.metrics = scheduler, target, draft, metrics

    @torch.inference_mode()
    def prefill(self, seqs):
        seq = seqs[0]
        logits, features = self.target.call("run_dflash_target", seqs, True)
        seq.recovery_token_id = int(sample_probs(probabilities(logits.reshape(-1, logits.shape[-1])[-1], seq.temperature)).item())
        seq.num_cached_tokens = seq.num_tokens
        self.draft.prefill(seq, features)
        return seq.num_tokens

    @torch.inference_mode()
    def decode(self, seqs):
        seq = seqs[0]
        k = self.scheduler.candidate_count(seq)
        if k:
            candidates, q = self.draft.propose(seq, k)
        else:
            candidates = torch.empty(0, dtype=torch.long, device=self.draft.device)
            q = torch.empty(0, self.draft.vocab_size, device=self.draft.device)
        round_record = None
        if k and hasattr(self.draft, "prepare_branches"):
            # Intentionally before target verification: branch inputs cannot
            # include this round's accepted length, bonus, or target features.
            self.draft.prepare_branches(seq, candidates)
        anchor = seq.recovery_token_id
        old_len, old_last = seq.num_tokens, seq.last_token
        try:
            seq.append_token(anchor)
            for token in candidates.tolist():
                seq.append_token(token)
            start = perf_counter()
            logits, features = self.target.call("run_dflash_target", seqs, False, k)
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            self.metrics["target_verify_times"].append(perf_counter() - start)
            p = probabilities(logits.reshape(k + 1, -1), seq.temperature)
            accepted, bonus = verify_block(candidates, p, q)
        finally:
            del seq.token_ids[old_len:]
            seq.num_tokens, seq.last_token = old_len, old_last
        if k and hasattr(self.draft, "prepare_branches"):
            if hasattr(self.draft, "finish_branches"):
                self.draft.finish_branches()
            round_record = dict(self.draft.round_record)
        produced = self.scheduler.commit(seq, [anchor] + candidates[:accepted].tolist(), bonus)
        if seq.is_finished:
            self.draft.reset()
        else:
            self.draft.update(features, accepted)
        self.metrics["accepted_suffix_lens_with_recovery"].append(produced)
        if round_record is not None:
            round_record.update(accepted=accepted, emitted_tokens=produced,
                                selected_next_pos=None if seq.is_finished else accepted)
            self.metrics.setdefault("dflash_position_rounds", []).append(round_record)
        return produced
