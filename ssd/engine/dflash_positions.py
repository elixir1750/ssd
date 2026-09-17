"""All-acceptance-position DFlash, without enumerating the unknown bonus.

This is a serial execution reference for a future asynchronous runner. Every
branch is completed BEFORE the current target verification starts. It uses
one generation of draft self-KV as a proxy, never inserts it into target KV,
and never recomputes a saved proposal after the real bonus is revealed.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.nn import functional as F
from transformers import DynamicCache

from ssd.engine.dflash_sync import DFlashRunner
from ssd.engine.dflash_support import accepted_features, draft_value, probabilities, sample_probs
from ssd.models.dflash import rotate_half


@dataclass
class PositionProposal:
    seq_id: int
    start: int  # position of the future bonus / known anchor at consumption
    source_pos: int | None  # accepted length which selected this proposal
    tokens: torch.Tensor
    probs: torch.Tensor
    # Own-block KV only: anchor/unknown bonus + candidates; no historical KV.
    block_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]


def snapshot_kv(cache, start, end):
    return tuple((key[..., start:end, :].clone(), value[..., start:end, :].clone())
                 for key, value in cache)


@torch.inference_mode()
def append_target_features(model, cache, features):
    """Append exactly the target-derived KV that the ordinary draft forward uses.

    These projections are independent of draft queries. This refreshes canonical
    context even when the next candidates came from a saved speculative branch.
    """
    start = cache.get_seq_length()
    fused = model.hidden_norm(model.fc(features))
    positions = torch.arange(start, start + features.shape[1], device=features.device)[None]
    cos, sin = model.rotary_emb(fused, positions)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        shape = (features.shape[0], features.shape[1], -1, attention.head_dim)
        key = attention.k_norm(attention.k_proj(fused).view(shape)).transpose(1, 2)
        value = attention.v_proj(fused).view(shape).transpose(1, 2)
        key = key * cos + rotate_half(key) * sin
        cache.update(key, value, index)


def branch_prefix_cache(context_cache, block_kv, keep_self_positions):
    """Share canonical context until update; concatenate only the visible proxy.

    The last known token is excluded and re-encoded as a clean query anchor.
    DynamicCache.update uses concatenation, so branches do not mutate shared KV.
    """
    if len(context_cache) != len(block_kv):
        raise ValueError("Context and proxy KV must have the same layers")
    if keep_self_positions < 0 or any(keep_self_positions > key.shape[-2] for key, _ in block_kv):
        raise ValueError("Proxy KV prefix is out of bounds")
    branch = DynamicCache()
    for index, ((key, value), (block_key, block_value)) in enumerate(zip(context_cache, block_kv)):
        if keep_self_positions:
            key = torch.cat((key, block_key[..., :keep_self_positions, :]), dim=-2)
            value = torch.cat((value, block_value[..., :keep_self_positions, :]), dim=-2)
        branch.update(key, value, index)
    return branch


class DFlashPositionsRunner(DFlashRunner):
    def __init__(self, config, device=None, reserve_target=True):
        super().__init__(config, device=device, reserve_target=reserve_target)
        dc, k = config.draft_hf_config, config.speculate_k
        element_size = next(self.model.parameters()).element_size()
        per_position = 2 * dc.num_hidden_layers * dc.num_key_value_heads * dc.head_dim
        # Peak includes a branch's working context and saved blocks/q for every k.
        # Account for temporary torch.cat copies while building a branch prefix.
        cache_bytes = per_position * (3 * config.max_model_len + (k + 1) * (k + 3)) * element_size
        proposal_bytes = (k + 2) * k * dc.vocab_size * 4  # q is always float32
        self._positions_reservation = (torch.empty(cache_bytes + proposal_bytes,
                                                   dtype=torch.uint8, device=self.device)
                                       if reserve_target else None)

    def bind_target(self, target):
        super().bind_target(target)
        self._positions_reservation = None

    def reset(self):
        super().reset()
        self.ready = None
        self.current = None
        self.branches = {}
        self.round_record = None
        if not hasattr(self, "branch_generator"):
            self.branch_generator = None

    def _generator(self):
        # Branch construction must not consume the RNG used by current verify.
        if self.branch_generator is None:
            self.branch_generator = torch.Generator(device=self.device)
            seed = (torch.initial_seed() + 0x5D_F1_A5) % (2**63 - 1)
            self.branch_generator.manual_seed(seed)
        return self.branch_generator

    def _refresh_context(self, seq):
        if self.seq_id != seq.seq_id or self.pending_features is None:
            raise RuntimeError("Missing accepted target features for this sequence")
        if self.cache.get_seq_length() + self.pending_features.shape[1] != seq.num_tokens:
            raise RuntimeError("Target feature delta does not match canonical draft context")
        append_target_features(self.model, self.cache, self.pending_features)
        self.pending_features = None

    def _forward_block(self, cache, ids, start):
        embeddings = F.embedding(ids, self.target.model.embed_tokens.weight)
        embeddings *= float(draft_value(self.model.config, "input_embedding_scale", 1.0))
        empty_features = embeddings.new_empty((1, 0, self.model.fc.in_features))
        return self.model(
            position_ids=torch.arange(start, start + ids.shape[1], device=self.device)[None],
            noise_embedding=embeddings, target_hidden=empty_features,
            past_key_values=cache, use_cache=True,
        )

    def _proposal(self, seq, hidden, cache, start, width, source_pos, generator=None):
        logits = self.model.compute_logits(hidden, lambda h: F.linear(h, self.target.lm_head.weight))
        temperature = seq.temperature if seq.draft_temperature is None else seq.draft_temperature
        q = probabilities(logits[0], temperature)
        tokens = sample_probs(q, generator)
        return PositionProposal(seq.seq_id, start, source_pos, tokens, q,
                                snapshot_kv(cache, start, start + width + 1))

    @torch.inference_mode()
    def propose(self, seq, k):
        self._refresh_context(seq)
        if self.ready is not None:
            proposal, self.ready = self.ready, None
            if (proposal.seq_id, proposal.start, proposal.tokens.numel()) != (seq.seq_id, seq.num_tokens, k):
                raise RuntimeError("Stale or incorrectly sized position proposal")
            # True bonus and fresh features deliberately do not change saved q.
            self.current = proposal
        else:
            ids = torch.full((1, k + 1), self.model.mask_token_id, dtype=torch.long, device=self.device)
            ids[0, 0] = seq.recovery_token_id
            hidden = self._forward_block(self.cache, ids, seq.num_tokens)
            self.current = self._proposal(seq, hidden[:, 1:], self.cache, seq.num_tokens, k, None)
            self.cache.crop(seq.num_tokens)
        return self.current.tokens, self.current.probs

    @torch.inference_mode()
    def prepare_branches(self, seq, candidates):
        k = candidates.numel()
        self.branches = {}
        if self.current is None or self.current.start != seq.num_tokens:
            raise RuntimeError("Position branches require the current proposal snapshot")
        known = [seq.recovery_token_id] + candidates.tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        widths = {}
        skipped = {}
        for accepted in range(k + 1):
            # Commit old anchor + accepted candidates; the new bonus is unknown.
            start = seq.num_tokens + 1 + accepted
            remaining = min(seq.max_new_tokens - seq.num_completion_tokens - 1 - accepted,
                            self.config.max_model_len - start)
            if not seq.ignore_eos and self.config.eos in known[:accepted + 1]:
                skipped[accepted] = "eos"
                continue
            if remaining <= 1:
                skipped[accepted] = "no_future_candidates"
                continue
            width = min(self.config.speculate_k, remaining - 1)
            # Keep proxy positions before the clean anchor at start-1.
            cache = branch_prefix_cache(self.cache, self.current.block_kv, accepted)
            if cache.get_seq_length() != start - 1:
                raise RuntimeError("Branch prefix has an incorrect logical length")
            ids = torch.full((1, width + 2), self.model.mask_token_id, dtype=torch.long, device=self.device)
            ids[0, 0] = known[accepted]
            # [clean anchor, unknown bonus MASK, candidate MASKs...]
            hidden = self._forward_block(cache, ids, start - 1)
            self.branches[accepted] = self._proposal(
                seq, hidden[:, 2:], cache, start, width, accepted, self._generator())
            widths[accepted] = width
            del hidden, cache
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.round_record = {
            "seq_id": seq.seq_id, "start": seq.num_tokens, "candidates": k,
            "source_pos": self.current.source_pos,
            "source": "bootstrap" if self.current.source_pos is None else "precomputed_position",
            "prepared_widths": widths, "skipped_positions": skipped,
            "prepare_ms": (perf_counter() - started) * 1000,
        }

    def update(self, features, accepted):
        self.pending_features = accepted_features(features, accepted)
        if accepted not in self.branches and self.round_record is not None:
            reason = self.round_record["skipped_positions"].get(accepted)
            if reason not in ("no_future_candidates", "eos"):
                raise RuntimeError("A required acceptance-position branch was not prepared")
        self.ready = self.branches.get(accepted)
        self.current = None
        self.branches = {}  # keep exactly one selected self-KV snapshot


def summarize_position_rounds(rounds):
    groups = {}
    for row in rounds:
        key = "bootstrap" if row["source_pos"] is None else str(row["source_pos"])
        stats = groups.setdefault(key, {"rounds": 0, "proposed": 0, "accepted": 0})
        stats["rounds"] += 1
        stats["proposed"] += row["candidates"]
        stats["accepted"] += row["accepted"]
    for stats in groups.values():
        stats["mean_accepted"] = stats["accepted"] / stats["rounds"]
        stats["accepted_fraction"] = stats["accepted"] / stats["proposed"] if stats["proposed"] else 0.0
    return groups
