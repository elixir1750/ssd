"""Spawned FDFlash protocol tests with real tiny models and CPU/Gloo."""
from collections import defaultdict
from pathlib import Path
import tempfile
import types
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import DynamicCache, Qwen3ForCausalLM
from test_dflash_cpu import config as tiny_config
from ssd.engine.dflash_async import DFlashAsyncClient, run_dflash_worker
from ssd.engine.dflash_positions import DFlashPositionsRunner
from ssd.engine.dflash_sync import DFlashScheduler, DFlashStep
from ssd.engine.sequence import Sequence
from ssd.models.dflash import DFlashDraftModel
from ssd.sampling_params import SamplingParams
from ssd.utils.distributed import init_model_parallel, create_async_group


def tiny_runner(config, device):
    runner = DFlashPositionsRunner.__new__(DFlashPositionsRunner)
    runner.config, runner.device = config, device
    with torch.random.fork_rng():
        torch.manual_seed(23)
        runner.model = DFlashDraftModel(tiny_config()).eval()
    runner.target = None
    runner.reset()
    if getattr(config, "prepare_started", None) is not None:
        original = runner.prepare_branches
        def gated(seq, tokens):
            config.prepare_started.set()
            if not config.prepare_release.wait(10):
                raise TimeoutError("Test preparation gate timed out")
            if getattr(config, "fail_prepare", False):
                raise RuntimeError("Injected draft preparation failure")
            original(seq, tokens)
        runner.prepare_branches = gated
    return runner


def cpu_worker(config, connection):
    # Import this test module first on spawn, installing the CPU package shim.
    run_dflash_worker(config, 1, connection, 17, "gloo", tiny_runner)


class AsyncChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.ctx = mp.get_context("spawn")
        self.directory = tempfile.TemporaryDirectory()
        target_config = tiny_config()
        target_config.num_hidden_layers = 4
        target_config.layer_types = ["full_attention"] * 4
        target_config.torch_dtype = torch.float32
        self.target = Qwen3ForCausalLM(target_config).eval()
        self.cfg = types.SimpleNamespace(
            num_gpus=2, distributed_timeout_seconds=20,
            distributed_init_method="file://" + str(Path(self.directory.name) / "store"),
            hf_config=target_config, dflash_target_layers=[0, 2], max_model_len=30,
            eos=16, speculate_k=3, num_kvcache_blocks=8, kvcache_block_size=8)
        Sequence.block_size = 8
        self.client = None
        self.process = None
        self.connection = None

    def launch(self, gated=False, fail=False):
        if gated:
            self.cfg.prepare_started = self.ctx.Event()
            self.cfg.prepare_release = self.ctx.Event()
            self.cfg.fail_prepare = fail
        parent, child = self.ctx.Pipe()
        self.connection = parent
        self.process = self.ctx.Process(target=cpu_worker, args=(self.cfg, child))
        self.process.start()
        child.close()
        init_model_parallel(self.cfg, 0, 1, torch.device("cpu"), "gloo")
        group = create_async_group(self.cfg)
        self.client = DFlashAsyncClient(self.cfg, parent, self.process, group, "cpu")
        self.client.bind_target(self.target)

    def tearDown(self):
        try:
            if self.client is not None:
                self.client.shutdown()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            if self.client is not None:
                self.client.join()
                self.assertFalse(self.process.is_alive())
            elif self.process is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=5)
            if self.connection is not None:
                self.connection.close()
            self.directory.cleanup()

    def sequence(self, prompt=None, length=12):
        seq = Sequence([1, 2] if prompt is None else prompt,
                       SamplingParams(temperature=0, max_new_tokens=length, ignore_eos=True))
        seq.recovery_token_id = 3
        return seq

    def test_config_accepts_two_gpu_positions_and_rejects_invalid_modes(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, SSD_HF_CACHE=self.directory.name, SSD_DATASET_DIR=self.directory.name):
            from ssd.config import Config
        root = Path(self.directory.name)
        target_path, draft_path = root / "target", root / "draft"
        self.cfg.hf_config.save_pretrained(target_path)
        tiny_config().save_pretrained(draft_path)
        args = dict(model=str(target_path), draft=str(draft_path), use_dflash=True,
                    dflash_all_positions=True, draft_async=True, num_gpus=2,
                    max_num_seqs=1, speculate_k=3, max_model_len=30)
        cfg = Config(**args)
        self.assertTrue(cfg.enforce_eager)
        self.assertIsNone(cfg.fan_out_list)
        for changes in [dict(num_gpus=1), dict(num_gpus=3), dict(dflash_all_positions=False),
                        dict(max_num_seqs=2), dict(kvcache_block_size=4)]:
            with self.assertRaises(ValueError):
                Config(**(args | changes))
        self.assertFalse(Config(**(args | dict(draft_async=False, num_gpus=1))).draft_async)

    @torch.inference_mode()
    def test_every_selected_branch_preserves_q_across_transport(self):
        self.launch()
        for temperature, accepted in [(t, k) for t in (0, 0.8) for k in range(4)]:
            seq = self.sequence()
            seq.temperature = temperature
            features = torch.randn(seq.num_tokens, 64)
            local = tiny_runner(self.cfg, torch.device("cpu"))
            local.bind_target(self.target)
            local.prefill(seq, features)
            self.client.prefill(seq, features)
            expected, q = local.propose(seq, 3)
            tokens, actual_q = self.client.propose(seq, 3)
            torch.testing.assert_close(actual_q, q, atol=0, rtol=0)
            if temperature == 0:
                torch.testing.assert_close(tokens, expected, atol=0, rtol=0)
            local.prepare_branches(seq, tokens)
            self.client.prepare_branches(seq, tokens)
            self.client.finish_branches()
            saved_q = local.branches[accepted].probs.clone()
            for token in [seq.recovery_token_id] + tokens[:accepted].tolist():
                seq.append_token(token)
            seq.recovery_token_id = 12
            delta = torch.randn(4, 64)
            local.update(delta, accepted)
            self.client.update(delta, accepted)
            expected, _ = local.propose(seq, 3)
            tokens, actual_q = self.client.propose(seq, 3)
            torch.testing.assert_close(actual_q, saved_q, atol=0, rtol=0)
            # The remote branch RNG persists across requests; the local fixture
            # is fresh. Only greedy tokens have an exact cross-request oracle.
            if temperature == 0:
                torch.testing.assert_close(tokens, expected, atol=0, rtol=0)
            self.client.reset()

    @torch.inference_mode()
    def test_target_can_run_while_branch_preparation_is_outstanding(self):
        self.launch(gated=True)
        seq = self.sequence()
        self.client.prefill(seq, torch.randn(2, 64))
        tokens, _ = self.client.propose(seq, 3)
        self.client.prepare_branches(seq, tokens)
        self.assertTrue(self.cfg.prepare_started.wait(5))
        # Worker is blocked until AFTER this real target forward completes.
        result = self.target(torch.tensor([[1, 2, 3] + tokens.tolist()]), use_cache=False)
        self.assertEqual(result.logits.shape, (1, 6, 17))
        self.assertTrue(self.client._preparing)
        self.cfg.prepare_release.set()
        self.client.finish_branches()
        self.assertIn("branch_wait_ms", self.client.round_record)
        self.client.reset()

    @torch.inference_mode()
    def test_real_hf_target_multiple_requests_and_limits(self):
        self.launch()
        cfg, target = self.cfg, self.target
        scheduler = DFlashScheduler(cfg)
        class Target:
            def call(self, method, seqs, prefill, k=0):
                seq = seqs[0]
                if prefill:
                    self.cache = DynamicCache()
                    ids = seq.token_ids
                else:
                    self.cache.crop(seq.num_cached_tokens)
                    ids = seq.token_ids[-(k + 1):]
                result = target(torch.tensor([ids]), past_key_values=self.cache,
                                use_cache=True, output_hidden_states=True)
                features = torch.cat([result.hidden_states[i + 1] for i in cfg.dflash_target_layers], -1)[0]
                return (result.logits[0, -1:] if prefill else result.logits[0]), features
        for prompt, length in [([1, 2], 12), ([4, 7, 2], 9), ([3], 1), ([1] * 28, 6)]:
            expected = []
            for _ in range(min(length, cfg.max_model_len - len(prompt))):
                logits = target(torch.tensor([prompt + expected]), use_cache=False).logits
                expected.append(int(logits[0, -1].argmax()))
            seq = self.sequence(prompt, length)
            scheduler.add(seq)
            metrics = defaultdict(list)
            step = DFlashStep(scheduler, Target(), self.client, metrics)
            for _ in range(length + 2):
                if scheduler.is_finished():
                    break
                seqs, prefill = scheduler.schedule()
                step.prefill(seqs) if prefill else step.decode(seqs)
            self.assertTrue(scheduler.is_finished())
            self.assertEqual(seq.completion_token_ids, expected)
            self.assertFalse(scheduler.block_manager.used_block_ids)
            records = metrics["dflash_position_rounds"]
            for previous, current in zip(records, records[1:]):
                self.assertEqual(current["source_pos"], previous["accepted"])

    @torch.inference_mode()
    def test_eos_resets_remote_state_before_next_request(self):
        self.launch()
        scheduler = DFlashScheduler(self.cfg)
        class Target:
            def call(self, method, seqs, prefill, k=0):
                seq = seqs[0]
                ids = seq.token_ids if prefill else seq.token_ids[-(k + 1):]
                logits = torch.full((len(ids), 17), -100.)
                for i, token in enumerate(ids):
                    logits[i, (token + 1) % 17] = 100.
                return (logits[-1:] if prefill else logits), torch.ones(len(ids), 64)
        for _ in range(2):
            seq = self.sequence([14], 7)
            seq.ignore_eos = False
            scheduler.add(seq)
            step = DFlashStep(scheduler, Target(), self.client, defaultdict(list))
            for _ in range(10):
                if scheduler.is_finished():
                    break
                seqs, prefill = scheduler.schedule()
                step.prefill(seqs) if prefill else step.decode(seqs)
            self.assertTrue(scheduler.is_finished())
            self.assertEqual(seq.completion_token_ids, [15, 16])
            self.assertFalse(scheduler.block_manager.used_block_ids)

    @torch.inference_mode()
    def test_draft_failure_is_propagated_and_worker_reaped(self):
        self.launch(gated=True, fail=True)
        seq = self.sequence()
        self.client.prefill(seq, torch.randn(2, 64))
        tokens, _ = self.client.propose(seq, 3)
        self.client.prepare_branches(seq, tokens)
        self.cfg.prepare_release.set()
        with self.assertRaisesRegex(RuntimeError, "Injected draft preparation failure"):
            self.client.finish_branches()

    @torch.inference_mode()
    def test_stale_round_is_rejected(self):
        self.launch()
        seq = self.sequence()
        self.client.prefill(seq, torch.randn(2, 64))
        tokens, _ = self.client.propose(seq, 3)
        seq.append_token(4)
        self.client.prepare_branches(seq, tokens)
        with self.assertRaisesRegex(RuntimeError, "Stale FDFlash preparation"):
            self.client.finish_branches()

    @torch.inference_mode()
    def test_timeout_stops_only_owned_worker(self):
        self.launch(gated=True)
        seq = self.sequence()
        self.client.prefill(seq, torch.randn(2, 64))
        tokens, _ = self.client.propose(seq, 3)
        self.client.prepare_branches(seq, tokens)
        self.assertTrue(self.cfg.prepare_started.wait(5))
        self.client.timeout = 0.1
        with self.assertRaises(TimeoutError):
            self.client.finish_branches()


if __name__ == "__main__":
    unittest.main()
