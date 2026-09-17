"""Two-GPU FDFlash on SSD's rank-0 target / rank-1 draft process groups.

The pipe carries bounded control messages; model tensors stay on the device
and use SSD's NCCL async group. PREPARE has a split start/finish so target
verification executes while the draft process prepares acceptance positions.
"""
from __future__ import annotations

from time import monotonic, perf_counter
from types import SimpleNamespace
import traceback

import torch
import torch.distributed as dist

from ssd.engine.dflash_positions import DFlashPositionsRunner
from ssd.utils.distributed import init_model_parallel, create_async_group
from ssd.utils.async_helpers.nccl_pack import send_int64, recv_int64


class DFlashAsyncClient:
    """Target-side adapter for the existing DFlashStep runner interface."""

    def __init__(self, config, connection, process, group, device):
        self.config, self.connection, self.process = config, connection, process
        self.group, self.device = group, torch.device(device)
        self.dtype = config.hf_config.torch_dtype or torch.bfloat16
        self.vocab_size = config.hf_config.vocab_size
        self.timeout = config.distributed_timeout_seconds
        self._serial = 0
        self._pending = None
        self._broken = False
        self._closed = False
        self._preparing = False
        self.round_record = None

    def _start(self, op, **data):
        if self._closed or self._broken:
            raise RuntimeError("FDFlash worker is closed or failed")
        if self._pending is not None:
            raise RuntimeError("Previous FDFlash command is still outstanding")
        self._serial += 1
        self._pending = self._serial
        try:
            self.connection.send(dict(id=self._serial, op=op, **data))
        except (BrokenPipeError, EOFError, OSError) as error:
            self._broken = True
            raise RuntimeError("FDFlash worker control channel failed") from error

    def _reply(self, stage="done"):
        deadline = monotonic() + self.timeout
        try:
            while not self.connection.poll(min(0.1, max(0, deadline - monotonic()))):
                if not self.process.is_alive():
                    raise RuntimeError(f"FDFlash worker exited with code {self.process.exitcode}")
                if monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for FDFlash worker")
            reply = self.connection.recv()
            if reply.get("error"):
                raise RuntimeError(f"FDFlash worker failed:\n{reply['error']}")
            if reply.get("id") != self._pending or reply.get("stage") != stage:
                raise RuntimeError("Stale or out-of-order FDFlash worker response")
            if stage == "done":
                self._pending = None
            return reply.get("value")
        except BaseException:
            self._broken = True
            raise

    def _send_tensor(self, tensor):
        try:
            dist.send(tensor.contiguous(), dst=1, group=self.group)
        except BaseException:
            self._broken = True
            raise

    @torch.inference_mode()
    def bind_target(self, target):
        self._start("bind", eos=self.config.eos)
        self._reply("ready")
        self._send_tensor(target.model.embed_tokens.weight)
        if not self.config.hf_config.tie_word_embeddings:
            self._send_tensor(target.lm_head.weight)
        self._reply()

    def _features(self, op, features, **data):
        if features.ndim != 2 or features.shape[1] != (
                self.config.hf_config.hidden_size * len(self.config.dflash_target_layers)):
            raise ValueError("Invalid FDFlash target feature shape")
        self._start(op, rows=features.shape[0], **data)
        self._reply("ready")
        self._send_tensor(features.to(device=self.device, dtype=self.dtype))
        self._reply()

    def prefill(self, seq, features):
        self._features("prefill", features, seq=seq)

    def propose(self, seq, k):
        self._start("propose", seq=seq, k=k)
        self._reply("tensors")
        try:
            tokens = recv_int64(self.group, 1, k, self.device)
            q = torch.empty((k, self.config.hf_config.vocab_size), device=self.device, dtype=torch.float32)
            dist.recv(q, src=1, group=self.group)
        except BaseException:
            self._broken = True
            raise
        self._reply()
        return tokens, q

    def prepare_branches(self, seq, candidates):
        # No receive here: returning immediately is the overlap boundary.
        self._start("prepare", seq_id=seq.seq_id, start=seq.num_tokens, k=candidates.numel())
        self._preparing = True

    def finish_branches(self):
        if not self._preparing:
            raise RuntimeError("No FDFlash branch preparation is outstanding")
        start = perf_counter()
        self.round_record = self._reply()
        self.round_record["branch_wait_ms"] = (perf_counter() - start) * 1000
        self._preparing = False

    def update(self, features, accepted):
        self._features("update", features[:accepted + 1], accepted=accepted)

    def reset(self):
        if self._preparing:
            self.finish_branches()
        self._start("reset")
        self._reply()
        self.round_record = None

    def shutdown(self):
        if self._closed:
            return
        try:
            if not self._broken:
                if self._preparing:
                    self.finish_branches()
                self._start("stop")
                self._reply()
        except BaseException:
            self._broken = True
            raise
        finally:
            self._closed = True
            self.connection.close()
            if self._broken and self.process.is_alive():
                self.process.terminate()

    def join(self):
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        if self.process.exitcode not in (0, None) and not self._broken:
            raise RuntimeError(f"FDFlash worker exited with code {self.process.exitcode}")


def serve_dflash(config, runner, connection, group, device):
    """Ordered protocol, also exercised with real draft layers over CPU/Gloo."""
    dtype = config.hf_config.torch_dtype or torch.bfloat16
    feature_dim = config.hf_config.hidden_size * len(config.dflash_target_layers)
    state, seq, candidates = "unbound", None, None
    serial = 0
    message = {}

    def reply(stage="done", value=None):
        connection.send(dict(id=message["id"], stage=stage, value=value))

    def receive_features(rows):
        if not 0 < rows <= config.max_model_len:
            raise ValueError("Invalid FDFlash feature length")
        reply("ready")
        features = torch.empty((rows, feature_dim), dtype=dtype, device=device)
        dist.recv(features, src=0, group=group)
        return features

    try:
        while True:
            message = connection.recv()
            if message["id"] != serial + 1:
                raise RuntimeError("Stale FDFlash command")
            serial = message["id"]
            op = message["op"]
            if op == "stop":
                runner.reset()
                reply()
                return
            if op == "bind" and state == "unbound":
                config.eos = message["eos"]
                shape = (config.hf_config.vocab_size, config.hf_config.hidden_size)
                embedding = torch.empty(shape, dtype=dtype, device=device)
                head = embedding if config.hf_config.tie_word_embeddings else torch.empty_like(embedding)
                reply("ready")
                dist.recv(embedding, src=0, group=group)
                if head is not embedding:
                    dist.recv(head, src=0, group=group)
                runner.bind_target(SimpleNamespace(
                    model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embedding)),
                    lm_head=SimpleNamespace(weight=head)))
                state = "idle"
            elif op == "prefill" and state == "idle":
                seq = message["seq"]
                if message["rows"] != seq.num_tokens:
                    raise ValueError("Prefill features must cover the prompt")
                runner.prefill(seq, receive_features(message["rows"]))
                state = "prefilled"
            elif op == "propose" and state in ("prefilled", "updated"):
                seq, k = message["seq"], message["k"]
                if not 0 < k <= config.speculate_k:
                    raise ValueError("Invalid candidate count")
                candidates, q = runner.propose(seq, k)
                reply("tensors")
                send_int64(group, 0, candidates)
                dist.send(q.contiguous(), dst=0, group=group)
                state = "proposed"
            elif op == "prepare" and state == "proposed":
                if (message["seq_id"], message["start"], message["k"]) != (
                        seq.seq_id, seq.num_tokens, candidates.numel()):
                    raise RuntimeError("Stale FDFlash preparation request")
                runner.prepare_branches(seq, candidates)
                state = "prepared"
                reply(value=runner.round_record)
                continue
            elif op == "update" and state == "prepared":
                accepted = message["accepted"]
                if not 0 <= accepted <= candidates.numel() or message["rows"] != accepted + 1:
                    raise ValueError("Invalid accepted-feature delta")
                runner.update(receive_features(message["rows"]), accepted)
                state = "updated"
            elif op == "reset" and state != "unbound":
                runner.reset()
                seq, candidates = None, None
                state = "idle"
            else:
                raise RuntimeError(f"Unexpected FDFlash command {op!r} in state {state!r}")
            reply()
    except Exception:
        try:
            connection.send(dict(id=message.get("id", 0), error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


def run_dflash_worker(config, rank, connection, seed, backend="nccl", runner_factory=None):
    """Spawn entry point matching ModelRunner's group-creation order."""
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    try:
        if backend == "nccl":
            torch.cuda.set_device(device)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        init_model_parallel(config, rank, 1, device, backend)
        runner = (DFlashPositionsRunner(config, device=device, reserve_target=False)
                  if runner_factory is None else runner_factory(config, device))
        group = create_async_group(config)
        with torch.inference_mode():
            serve_dflash(config, runner, connection, group, device)
    except Exception:
        # Includes failures before serve_dflash starts (checkpoint/group setup).
        try:
            connection.send(dict(id=0, error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        if dist.is_initialized():
            dist.destroy_process_group()
