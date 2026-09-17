"""CPU checks without importing SSD's CUDA-only package initializer.

Run in a fresh process: python -m unittest discover -s tests -p test_dflash_cpu.py
These tests exercise real draft attention; they do not emulate native CUDA kernels.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import tempfile
import unittest

import torch
from torch import nn
from transformers import DynamicCache, Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
for name in ("ssd", "ssd.engine", "ssd.models", "ssd.utils"):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules[name] = module

from ssd.engine.dflash_support import (
    accepted_features, probabilities, sample_probs, validate_dflash_config, verify_block,
)
from ssd.models.dflash import DFlashDraftModel
from ssd.engine.dflash_sync import DFlashRunner, DFlashScheduler, DFlashStep
from ssd.engine.sequence import Sequence
from ssd.sampling_params import SamplingParams


def config():
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      vocab_size=17, max_position_embeddings=128)
    cfg.num_target_layers = 4
    cfg.dflash_config = {"target_layer_ids": [0, 2], "mask_token_id": 16, "block_size": 4}
    return cfg


class DFlashCPUChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_config_rejects_mismatched_or_correlated_models(self):
        draft = config()
        target = config()
        target.num_hidden_layers = 4
        self.assertEqual(validate_dflash_config(target, draft), [0, 2])
        draft.architectures = ["DFlash2DraftModel"]
        with self.assertRaises(ValueError):
            validate_dflash_config(target, draft)
        draft.architectures = ["DFlashDraftModel"]
        draft.num_target_layers = 8
        with self.assertRaises(ValueError):
            validate_dflash_config(target, draft)

    @unittest.skipUnless(os.environ.get("DFLASH_REFERENCE_MODEL"), "Set DFLASH_REFERENCE_MODEL to upstream dflash/model.py")
    @torch.inference_mode()
    def test_matches_pinned_upstream_forward(self):
        spec = importlib.util.spec_from_file_location("dflash_upstream_reference", os.environ["DFLASH_REFERENCE_MODEL"])
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream
        spec.loader.exec_module(upstream)
        ours = DFlashDraftModel(config()).eval()
        reference = upstream.DFlashDraftModel(config()).eval()
        reference.load_state_dict(ours.state_dict(), strict=True)
        ca, cb = DynamicCache(), DynamicCache()
        for cached, new, width in [(0, 5, 4), (5, 2, 4), (7, 1, 2)]:
            args = dict(position_ids=torch.arange(cached, cached + new + width)[None],
                        target_hidden=torch.randn(1, new, 64),
                        noise_embedding=torch.randn(1, width, 32), use_cache=True)
            torch.testing.assert_close(ours(past_key_values=ca, **args),
                                       reference(past_key_values=cb, **args), atol=0, rtol=0)
            ca.crop(cached + new)
            cb.crop(cached + new)

    def test_original_checkpoint_roundtrip(self):
        model = DFlashDraftModel(config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored, info = DFlashDraftModel.from_pretrained(directory, output_loading_info=True)
            self.assertFalse(any(info.values()), info)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)

    @torch.inference_mode()
    def test_incremental_target_kv_matches_full_context(self):
        model = DFlashDraftModel(config()).eval()
        features = torch.randn(1, 7, 64)
        old_noise, new_noise = torch.randn(1, 4, 32), torch.randn(1, 4, 32)
        cache = DynamicCache()
        model(position_ids=torch.arange(9)[None], target_hidden=features[:, :5],
              noise_embedding=old_noise, past_key_values=cache, use_cache=True)
        cache.crop(5)
        incremental = model(position_ids=torch.arange(5, 11)[None], target_hidden=features[:, 5:],
                            noise_embedding=new_noise, past_key_values=cache, use_cache=True)
        full = model(position_ids=torch.arange(11)[None], target_hidden=features,
                     noise_embedding=new_noise, past_key_values=DynamicCache(), use_cache=True)
        torch.testing.assert_close(incremental, full, atol=1e-6, rtol=1e-5)
        cache.crop(7)
        self.assertEqual(cache.get_seq_length(), 7)

    def test_feature_slice_excludes_rejected_tokens_and_bonus(self):
        features = torch.arange(20).view(5, 4)
        for k in range(5):
            selected = accepted_features(features, k)
            self.assertEqual(selected.shape, (1, k + 1, 4))
            torch.testing.assert_close(selected[0], features[:k + 1])
            self.assertNotEqual(selected.data_ptr(), features.data_ptr())

    def test_zero_and_full_acceptance_and_no_candidates(self):
        p = torch.eye(3)[torch.tensor([0, 1, 2])]
        q = p[:2].clone()
        self.assertEqual(verify_block(torch.tensor([0, 1]), p, q), (2, 2))
        self.assertEqual(verify_block(torch.tensor([1]), torch.tensor([[1., 0.], [0., 1.]]),
                                      torch.tensor([[0., 1.]])), (0, 0))
        self.assertEqual(verify_block(torch.empty(0, dtype=torch.long), torch.tensor([[0., 1.]]),
                                      torch.empty(0, 2)), (0, 1))

    def test_sampling_recovers_target_distribution(self):
        # p != q exercises residual correction, including deterministic q.
        p = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
        for q in (torch.tensor([[0.9, 0.1]]), torch.tensor([[1., 0.]])):
            histogram = torch.zeros(2)
            gen = torch.Generator().manual_seed(813)
            for _ in range(12000):
                candidate = sample_probs(q, gen).flatten()
                k, bonus = verify_block(candidate, p, q, gen)
                histogram[int(candidate[0]) if k else bonus] += 1
            self.assertLess(abs(float(histogram[0] / histogram.sum()) - 0.6), 0.018)

    def test_native_feature_extraction_handles_delayed_residual(self):
        # Execute the actual modified Qwen3Model.forward with transparent layers.
        tree = ast.parse((ROOT / "ssd/models/qwen3.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3Model")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), "qwen3.py", "exec"), namespace)
        class Layer(nn.Module):
            def forward(self, positions, hidden, residual):
                complete = hidden if residual is None else hidden + residual
                return complete * 2, complete
        class Norm(nn.Module):
            def forward(self, hidden, residual):
                return (hidden + residual) / 2, hidden + residual
        instance = types.SimpleNamespace(embed_tokens=lambda ids: ids.float()[:, None],
                                         layers=[Layer(), Layer(), Layer()], norm=Norm())
        output, selected = namespace["forward"](instance, torch.tensor([1, 2]), None, [1, 0, 2])
        torch.testing.assert_close(selected, torch.tensor([[9., 3., 13.5], [18., 6., 27.]]))
        torch.testing.assert_close(output, selected[:, -1:])

    @torch.inference_mode()
    def test_sync_loop_matches_autoregressive_and_resets_between_requests(self):
        Sequence.block_size = 8
        cfg = types.SimpleNamespace(max_model_len=30, eos=16, num_kvcache_blocks=8,
                                    kvcache_block_size=8, speculate_k=3)
        scheduler = DFlashScheduler(cfg)
        draft = DFlashRunner.__new__(DFlashRunner)
        draft.device = torch.device("cpu")
        draft.model = DFlashDraftModel(config()).eval()
        target_weights = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=nn.Embedding(17, 32)), lm_head=nn.Linear(32, 17, bias=False))
        draft.target = target_weights
        draft.reset()

        class Target:
            def call(self, method, seqs, prefill, k=0):
                seq = seqs[0]
                ids = seq.token_ids if prefill else seq.token_ids[-(k + 1):]
                logits = torch.full((len(ids), 17), -100.)
                for i, token in enumerate(ids):
                    logits[i, (token + 1) % 17] = 100.
                features = torch.arange(len(ids) * 64).view(len(ids), 64).float() / 100
                return (logits[-1:] if prefill else logits), features

        for prompt, max_new, ignore_eos in [([1, 2], 9, True), ([14], 9, False), ([3], 1, True),
                                             ([1] * 28, 6, True), ([1, 2], 4, True)]:
            seq = Sequence(prompt, SamplingParams(temperature=0, max_new_tokens=max_new, ignore_eos=ignore_eos))
            scheduler.add(seq)
            metrics = defaultdict(list)
            step = DFlashStep(scheduler, Target(), draft, metrics)
            for _ in range(40):
                if scheduler.is_finished():
                    break
                seqs, prefill = scheduler.schedule()
                step.prefill(seqs) if prefill else step.decode(seqs)
            self.assertTrue(scheduler.is_finished())
            expected, token = [], prompt[-1]
            for _ in range(min(max_new, 30 - len(prompt))):
                token = (token + 1) % 17
                expected.append(token)
                if not ignore_eos and token == 16:
                    break
            self.assertEqual(seq.completion_token_ids, expected)
            self.assertEqual(draft.cache.get_seq_length(), 0)
            self.assertFalse(scheduler.block_manager.used_block_ids)


if __name__ == "__main__":
    unittest.main()
