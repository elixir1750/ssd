"""Validate and benchmark ordinary DFlash, serial FDFlash and two-GPU SSD."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_mode(args):
    import torch
    from ssd import LLM, SamplingParams
    if torch.cuda.device_count() < (2 if args.mode == "async" else 1):
        raise RuntimeError("Not enough visible CUDA GPUs")
    torch.manual_seed(0)
    report = {"status": "RUNNING", "mode": args.mode, "block_size": args.block_size,
              "target": args.target, "draft": args.draft, "cases": [],
              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "torch": torch.__version__, "cuda": torch.version.cuda, "performance": [],
              "timing_note": "Performance uses fixed 128-token greedy outputs with ignore_eos=True, one full-length warmup per prompt, then three repeats. Request wall time includes prefill, IPC, verification, drafting, sampling and reset; initialization and explicit warmups are separate."}
    engine = None
    try:
        start = perf_counter()
        engine = LLM(args.target, draft=args.draft, use_dflash=True, dflash_all_positions=args.mode != "dflash",
                     draft_async=args.mode == "async", num_gpus=2 if args.mode == "async" else 1,
                     speculate_k=args.block_size - 1, max_num_seqs=1, max_model_len=512,
                     max_num_batched_tokens=512, gpu_memory_utilization=0.3)
        report["initialization_seconds"] = perf_counter() - start
        tokenizer = engine.tokenizer
        sky = tokenizer.encode("Explain why the sky is blue in two sentences.")
        repeated = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 60)
        start = perf_counter()
        engine.generate([sky], SamplingParams(temperature=0, max_new_tokens=8),
                        use_tqdm=False, stream_callback=lambda *_: None)
        torch.cuda.synchronize(0)
        report["explicit_warmup_seconds"] = perf_counter() - start
        cases = [("sky", sky, 64),
                 ("code", tokenizer.encode("def fibonacci(n):\n"), 32),
                 ("page-crossing", repeated[:250], 32),
                 ("one-token", sky, 1),
                 ("context-limit", repeated[:508], 4)]
        for temperature, draft_temperature in [(0, 0), (0.8, 0.8), (0.8, 0)]:
            for name, ids, length in cases:
                print(f"{args.mode}: {name} T={temperature} draft_T={draft_temperature}", flush=True)
                start = perf_counter()
                outputs, metrics = engine.generate([ids], SamplingParams(
                    temperature=temperature, draft_temperature=draft_temperature, max_new_tokens=length),
                    use_tqdm=False, stream_callback=lambda *_: None)
                torch.cuda.synchronize(0)
                if not engine.is_finished() or engine.scheduler.block_manager.used_block_ids:
                    raise RuntimeError("Request did not finish or leaked target KV blocks")
                if len(outputs) != 1 or not 0 < len(outputs[0]["token_ids"]) <= min(length, 512 - len(ids)):
                    raise RuntimeError("Invalid generation length")
                if args.mode != "async" and (engine.dflash_runner.cache.get_seq_length() or
                        getattr(engine.dflash_runner, "ready", None) is not None):
                    raise RuntimeError("Draft state was not reset")
                if args.mode == "async" and not engine.draft_ps.is_alive():
                    raise RuntimeError("Draft worker exited before shutdown")
                report["cases"].append(dict(case=name, temperature=temperature,
                    draft_temperature=draft_temperature, output=outputs[0],
                    seconds=perf_counter() - start, metrics=deepcopy(metrics)))
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        for name, ids in [("sky", sky), ("code", tokenizer.encode("def fibonacci(n):\n")),
                          ("page-crossing", repeated[:250])]:
            params = SamplingParams(temperature=0, draft_temperature=0, max_new_tokens=128, ignore_eos=True)
            torch.cuda.synchronize(0)
            warm_start = perf_counter()
            engine.generate([ids], params, use_tqdm=False, stream_callback=lambda *_: None)
            torch.cuda.synchronize(0)
            entry = dict(case=name, prompt_tokens=len(ids), output_tokens=128,
                         warmup_seconds=perf_counter() - warm_start, repeats=[])
            report["performance"].append(entry)
            for repeat in range(3):
                torch.cuda.synchronize(0)
                start = perf_counter()
                outputs, metrics = engine.generate([ids], params, use_tqdm=False,
                                                   stream_callback=lambda *_: None)
                torch.cuda.synchronize(0)
                seconds = perf_counter() - start
                if len(outputs[0]["token_ids"]) != 128 or not engine.is_finished():
                    raise RuntimeError("Performance request did not generate exactly 128 tokens")
                if engine.scheduler.block_manager.used_block_ids:
                    raise RuntimeError("Performance request leaked target KV blocks")
                rounds = metrics["accepted_suffix_lens_with_recovery"]
                positions = metrics.get("dflash_position_rounds", [])
                entry["repeats"].append(dict(seconds=seconds, tokens_per_second=128 / seconds,
                    tokens=outputs[0]["token_ids"], mean_committed_per_round=sum(rounds) / len(rounds),
                    target_verify_seconds=sum(metrics["target_verify_times"]),
                    branch_prepare_seconds=sum(r["prepare_ms"] for r in positions) / 1000,
                    branch_wait_seconds=sum(r.get("branch_wait_ms", 0) for r in positions) / 1000))
                print(f"PERF {args.mode} {name} repeat={repeat}: {128 / seconds:.2f} token/s", flush=True)
            entry["median_seconds"] = median(r["seconds"] for r in entry["repeats"])
            entry["median_tokens_per_second"] = 128 / entry["median_seconds"]
            entry["repeat_outputs_equal"] = all(r["tokens"] == entry["repeats"][0]["tokens"] for r in entry["repeats"])
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["aggregate_tokens_per_second"] = 384 / sum(r["median_seconds"] for r in report["performance"])
        report["status"] = "PASS"
    except Exception as error:
        report["status"], report["error"] = "ERROR", f"{type(error).__name__}: {error}"
    finally:
        if engine is not None:
            try:
                engine.exit(hard=False)
                if args.mode == "async":
                    report["worker_exitcode"] = engine.draft_ps.exitcode
                    if engine.draft_ps.is_alive() or engine.draft_ps.exitcode != 0:
                        raise RuntimeError("Draft worker did not shut down cleanly")
            except Exception as error:
                report["status"], report["shutdown_error"] = "ERROR", f"{type(error).__name__}: {error}"
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("dflash", "serial", "async"))
    args = parser.parse_args()
    if not 2 <= args.block_size <= 128:
        parser.error("block-size must be 2..128 for the default KV page size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result")
    if args.mode:
        return run_mode(args)
    report = {"status": "RUNNING", "modes": {}, "greedy_comparisons": [],
              "scope": "Serial FDFlash vs two-GPU FDFlash; not native AR/HF parity or stochastic distributional equivalence."}
    try:
        for mode in ("dflash", "serial", "async"):
            output = args.output.with_name(args.output.stem + "-" + mode + ".json")
            command = [sys.executable, "-O", str(Path(__file__).resolve()),
                       "--target", args.target, "--draft", args.draft,
                       "--block-size", str(args.block_size), "--output", str(output), "--mode", mode]
            result = subprocess.run(command, check=False)
            report["modes"][mode] = dict(exitcode=result.returncode, output=str(output))
            if result.returncode != 0:
                raise RuntimeError(f"{mode} verification failed; inspect {output}")
        serial = json.loads(Path(report["modes"]["serial"]["output"]).read_text())
        parallel = json.loads(Path(report["modes"]["async"]["output"]).read_text())
        for left, right in zip(serial["cases"], parallel["cases"], strict=True):
            if left["temperature"] != 0:
                continue
            a, b = left["output"]["token_ids"], right["output"]["token_ids"]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if first is None and len(a) != len(b):
                first = min(len(a), len(b))
            report["greedy_comparisons"].append(dict(case=left["case"], equal=a == b, first_divergence=first))
        baseline = json.loads(Path(report["modes"]["dflash"]["output"]).read_text())
        report["performance"] = {}
        for label, data in [("dflash", baseline), ("serial", serial), ("async", parallel)]:
            throughput = data["aggregate_tokens_per_second"]
            report["performance"][label] = dict(tokens_per_second=throughput,
                speedup_vs_single_dflash=throughput / baseline["aggregate_tokens_per_second"],
                gpus=2 if label == "async" else 1,
                initialization_seconds=data["initialization_seconds"],
                explicit_warmup_seconds=data["explicit_warmup_seconds"] + sum(c["warmup_seconds"] for c in data["performance"]))
        report["performance_output_parity"] = [dict(case=a["case"],
            serial_matches_single=a["repeats"][0]["tokens"] == b["repeats"][0]["tokens"],
            async_matches_single=a["repeats"][0]["tokens"] == c["repeats"][0]["tokens"])
            for a, b, c in zip(baseline["performance"], serial["performance"], parallel["performance"], strict=True)]
        passed = len(report["greedy_comparisons"]) == 5 and all(c["equal"] for c in report["greedy_comparisons"])
        report["status"] = "PASS" if passed else "FAIL"
    except Exception as error:
        report["status"], report["error"] = "ERROR", f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
