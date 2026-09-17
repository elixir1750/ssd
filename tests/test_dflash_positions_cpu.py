"""Real tiny-draft tests for the all-position, unknown-bonus experiment."""
from __future__ import annotations

from collections import defaultdict
import types
import unittest

import torch
from torch import nn
from transformers import DynamicCache

# Reuse the CPU import bootstrap, which bypasses SSD's CUDA-only initializer.
from test_dflash_cpu import config
from ssd.engine.dflash_positions import (
    DFlashPositionsRunner, append_target_features, summarize_position_rounds,
)
from ssd.engine.dflash_sync import DFlashScheduler, DFlashStep
from ssd.engine.sequence import Sequence
from ssd.models.dflash import DFlashDraftModel
from ssd.sampling_params import SamplingParams


def make_runner(max_new=24, temperature=0.8, max_model_len=40):
    Sequence.block_size = 8
    cfg = types.SimpleNamespace(max_model_len=max_model_len, eos=16, speculate_k=3,
                                num_kvcache_blocks=8, kvcache_block_size=8)
    runner = DFlashPositionsRunner.__new__(DFlashPositionsRunner)
    runner.config, runner.device = cfg, torch.device("cpu")
    runner.model = DFlashDraftModel(config()).eval()
    runner.target = types.SimpleNamespace(
        model=types.SimpleNamespace(embed_tokens=nn.Embedding(17, 32)),
        lm_head=nn.Linear(32, 17, bias=False))
    runner.reset()
    seq = Sequence([1, 2], SamplingParams(temperature=temperature, max_new_tokens=max_new, ignore_eos=True))
    seq.recovery_token_id = 3
    features = torch.randn(seq.num_tokens, 64)
    runner.prefill(seq, features)
    return runner, seq, features


class PositionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    @torch.inference_mode()
    def test_direct_feature_refresh_matches_official_forward_context(self):
        runner, seq, features = make_runner()
        projected = DynamicCache()
        append_target_features(runner.model, projected, features[None])
        extra = torch.randn(1, 2, 64)
        append_target_features(runner.model, projected, extra)
        full = torch.cat((features[None], extra), dim=1)
        expected = DynamicCache()
        runner.model(position_ids=torch.arange(8)[None], target_hidden=full,
                     noise_embedding=torch.randn(1, 4, 32), past_key_values=expected, use_cache=True)
        expected.crop(4)
        for a, b in zip(projected, expected):
            torch.testing.assert_close(a[0], b[0])
            torch.testing.assert_close(a[1], b[1])

    @torch.inference_mode()
    def test_all_positions_anchors_mask_slot_and_rng_isolation(self):
        runner, seq, _ = make_runner()
        tokens, _ = runner.propose(seq, 3)
        snapshots = [(k.clone(), v.clone()) for k, v in runner.cache]
        captures = []
        def observe(module, args, kwargs):
            captures.append((kwargs["position_ids"].clone(), kwargs["noise_embedding"].clone(),
                             kwargs["past_key_values"].get_seq_length(), kwargs["target_hidden"].shape[1]))
        hook = runner.model.register_forward_pre_hook(observe, with_kwargs=True)
        rng_state = torch.get_rng_state().clone()
        runner.prepare_branches(seq, tokens)
        hook.remove()
        torch.testing.assert_close(torch.get_rng_state(), rng_state, atol=0, rtol=0)
        self.assertEqual(set(runner.branches), {0, 1, 2, 3})
        self.assertEqual(len(captures), 4)
        known = [seq.recovery_token_id] + tokens.tolist()
        for accepted, (positions, embeddings, prefix_len, fresh_count) in enumerate(captures):
            start = seq.num_tokens + accepted
            self.assertEqual(prefix_len, start)
            self.assertEqual(fresh_count, 0)  # current target verification has not happened
            torch.testing.assert_close(positions, torch.arange(start, start + 5)[None])
            torch.testing.assert_close(embeddings[0, 0], runner.target.model.embed_tokens.weight[known[accepted]])
            mask_embedding = runner.target.model.embed_tokens.weight[runner.model.mask_token_id]
            torch.testing.assert_close(embeddings[0, 1:], mask_embedding.expand(4, -1))
            proposal = runner.branches[accepted]
            self.assertEqual(proposal.tokens.numel(), 3)
            self.assertEqual(proposal.block_kv[0][0].shape[-2], 4)
            self.assertEqual(proposal.start, start + 1)
        for original, current in zip(snapshots, runner.cache):
            torch.testing.assert_close(original[0], current[0], atol=0, rtol=0)
            torch.testing.assert_close(original[1], current[1], atol=0, rtol=0)

    @torch.inference_mode()
    def test_real_bonus_does_not_regenerate_tokens_or_q(self):
        for accepted in range(4):
            runner, seq, initial_features = make_runner()
            candidates, _ = runner.propose(seq, 3)
            runner.prepare_branches(seq, candidates)
            saved = runner.branches[accepted]
            saved_tokens, saved_q = saved.tokens.clone(), saved.probs.clone()
            for token in [seq.recovery_token_id] + candidates[:accepted].tolist():
                seq.append_token(token)
            seq.recovery_token_id = 12  # actual bonus revealed only now
            new_features = torch.randn(4, 64)
            runner.update(new_features, accepted)
            self.assertFalse(runner.branches)
            calls = []
            hook = runner.model.register_forward_pre_hook(lambda *args: calls.append(True))
            tokens, q = runner.propose(seq, 3)
            hook.remove()
            self.assertFalse(calls)  # no re-drafting conditioned on real bonus
            self.assertEqual(q.data_ptr(), saved.probs.data_ptr())
            torch.testing.assert_close(tokens, saved_tokens, atol=0, rtol=0)
            torch.testing.assert_close(q, saved_q, atol=0, rtol=0)
            self.assertEqual(runner.cache.get_seq_length(), seq.num_tokens)
            expected = DynamicCache()
            append_target_features(runner.model, expected,
                                   torch.cat((initial_features, new_features[:accepted + 1]))[None])
            for a, b in zip(runner.cache, expected):
                torch.testing.assert_close(a[0], b[0])
                torch.testing.assert_close(a[1], b[1])

    @torch.inference_mode()
    def test_branch_widths_follow_generation_context_and_eos_limits(self):
        runner, seq, _ = make_runner(max_new=5)
        candidates, _ = runner.propose(seq, 3)
        runner.prepare_branches(seq, candidates)
        self.assertEqual({k: v.tokens.numel() for k, v in runner.branches.items()}, {0: 3, 1: 2, 2: 1})
        self.assertEqual(runner.round_record["skipped_positions"], {3: "no_future_candidates"})
        runner, seq, _ = make_runner(max_model_len=7)
        candidates, _ = runner.propose(seq, 3)
        runner.prepare_branches(seq, candidates)
        self.assertEqual({k: v.tokens.numel() for k, v in runner.branches.items()}, {0: 3, 1: 2, 2: 1})
        seq.ignore_eos = False
        seq.recovery_token_id = runner.config.eos
        runner.prepare_branches(seq, candidates)
        self.assertFalse(runner.branches)

    @torch.inference_mode()
    def test_stale_ready_proposal_is_rejected(self):
        runner, seq, _ = make_runner()
        candidates, _ = runner.propose(seq, 3)
        runner.prepare_branches(seq, candidates)
        seq.append_token(seq.recovery_token_id)
        runner.update(torch.randn(4, 64), 0)
        runner.ready.start += 1
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            runner.propose(seq, 3)

    @torch.inference_mode()
    def test_end_to_end_prepares_before_verify_and_uses_selected_branch(self):
        runner, _, _ = make_runner(temperature=0)
        scheduler = DFlashScheduler(runner.config)
        events = []
        original_prepare = runner.prepare_branches
        def prepare(seq, tokens):
            events.append("prepare")
            original_prepare(seq, tokens)
        runner.prepare_branches = prepare

        class Target:
            def call(self, method, seqs, prefill, k=0):
                seq = seqs[0]
                if not prefill and k:
                    self_test.assertEqual(events[-1], "prepare")
                    events.append("verify")
                ids = seq.token_ids if prefill else seq.token_ids[-(k + 1):]
                logits = torch.full((len(ids), 17), -100.)
                for i, token in enumerate(ids):
                    logits[i, (token + 1) % 17] = 100.
                return (logits[-1:] if prefill else logits), torch.ones(len(ids), 64)

        self_test = self
        all_records = []
        for prompt, max_new, ignore_eos in [([1, 2], 9, True), ([14], 7, False),
                                             ([1], 1, True), ([1] * 38, 7, True)]:
            seq = Sequence(prompt, SamplingParams(temperature=0, max_new_tokens=max_new, ignore_eos=ignore_eos))
            scheduler.add(seq)
            metrics = defaultdict(list)
            step = DFlashStep(scheduler, Target(), runner, metrics)
            for _ in range(50):
                if scheduler.is_finished():
                    break
                seqs, prefill = scheduler.schedule()
                step.prefill(seqs) if prefill else step.decode(seqs)
            self.assertTrue(scheduler.is_finished())
            expected, token = [], prompt[-1]
            for _ in range(min(max_new, 40 - len(prompt))):
                token = (token + 1) % 17
                expected.append(token)
                if not ignore_eos and token == 16:
                    break
            self.assertEqual(seq.completion_token_ids, expected)
            self.assertIsNone(runner.ready)
            self.assertFalse(runner.branches)
            self.assertEqual(runner.cache.get_seq_length(), 0)
            records = metrics["dflash_position_rounds"]
            for prev, current in zip(records, records[1:]):
                self.assertEqual(current["source"], "precomputed_position")
                self.assertEqual(current["source_pos"], prev["accepted"])
            all_records.extend(records)
        summary = summarize_position_rounds(all_records)
        self.assertEqual(sum(row["rounds"] for row in summary.values()), len(all_records))
        self.assertGreater(len(all_records), 2)


if __name__ == "__main__":
    unittest.main()
