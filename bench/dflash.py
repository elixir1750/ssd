"""Single-request native Qwen3 + original DFlash baseline and reference check."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Local dense Qwen3 checkpoint")
    parser.add_argument("--draft", required=True, help="Local original DFlash checkpoint")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--chat", action="store_true", help="Apply the target tokenizer's chat template")
    parser.add_argument("--block-size", type=int, default=16, help="Known anchor plus proposed tokens")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-reference", action="store_true",
                        help="Check native prefill features/logits and greedy output against HF Qwen3")
    parser.add_argument("--feature-atol", type=float, default=0.05)
    parser.add_argument("--feature-rtol", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.block_size < 2:
        parser.error("block-size must be at least 2 (one anchor plus a proposal)")
    if args.check_reference and args.temperature != 0:
        parser.error("--check-reference compares exact greedy tokens; use temperature=0")
    if not torch.cuda.is_available():
        parser.error("Native SSD requires CUDA; CPU-only checks live in tests/test_dflash_cpu.py")

    from ssd import LLM, SamplingParams
    from ssd.engine.dflash_support import validate_dflash_config

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    ids = (tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                         tokenize=True, add_generation_prompt=True)
           if args.chat else tokenizer.encode(args.prompt))
    target_config, draft_config = AutoConfig.from_pretrained(args.target), AutoConfig.from_pretrained(args.draft)
    layer_ids = validate_dflash_config(target_config, draft_config)
    limit = min(args.max_model_len, target_config.max_position_embeddings, draft_config.max_position_embeddings)
    if not ids or len(ids) >= limit or args.max_new_tokens < 1:
        parser.error("Need a nonempty prompt, positive max-new-tokens and free context space")

    reference = None
    if args.check_reference:
        target = AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=target_config.torch_dtype or torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        input_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            result = target(input_ids, output_hidden_states=True, use_cache=False)
            feature = torch.cat([result.hidden_states[i + 1] for i in layer_ids], dim=-1)[0].cpu()
            last_logits = result.logits[0, -1].cpu()
            del result
            generated = target.generate(
                input_ids, do_sample=False, max_new_tokens=min(args.max_new_tokens, limit - len(ids)),
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
            )[0, len(ids):].tolist()
        reference = {"feature": feature, "logits": last_logits, "tokens": generated}
        del target, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    engine = LLM(args.target, draft=args.draft, use_dflash=True,
                 speculate_k=args.block_size - 1, num_gpus=1, max_num_seqs=1,
                 max_model_len=limit, max_num_batched_tokens=max(16384, limit),
                 gpu_memory_utilization=args.memory_utilization)
    report = {"mode": "synchronous_dflash", "target": args.target, "draft": args.draft,
              "block_size": args.block_size, "target_layer_ids": layer_ids,
              "torch": torch.__version__}
    try:
        if reference is not None:
            engine.add_request(ids, SamplingParams(temperature=0, max_new_tokens=1))
            seqs, _ = engine.scheduler.schedule()
            with torch.inference_mode():
                logits, features = engine.model_runner.run_dflash_target(seqs, True)
            features, logits = features.cpu(), logits.reshape(-1, logits.shape[-1])[-1].cpu()
            report["feature_max_abs_error"] = float((features.float() - reference["feature"].float()).abs().max())
            report["logit_max_abs_error"] = float((logits.float() - reference["logits"].float()).abs().max())
            torch.testing.assert_close(features, reference["feature"], atol=args.feature_atol, rtol=args.feature_rtol)
            torch.testing.assert_close(logits, reference["logits"], atol=args.feature_atol, rtol=args.feature_rtol)
            engine.scheduler.block_manager.deallocate(seqs[0])
            engine.scheduler.running.clear()

        outputs, metrics = engine.generate([ids], SamplingParams(
            temperature=args.temperature, draft_temperature=args.draft_temperature,
            max_new_tokens=args.max_new_tokens), use_tqdm=False)
        torch.cuda.synchronize()
        report["outputs"] = outputs
        report["metrics"] = metrics
        if reference is not None:
            report["greedy_matches_hf"] = outputs[0]["token_ids"] == reference["tokens"]
            report["reference_tokens"] = reference["tokens"]
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        if reference is not None and not report["greedy_matches_hf"]:
            raise RuntimeError("Native DFlash output differs from HF greedy reference; inspect output report")
    finally:
        engine.exit(hard=False)


if __name__ == "__main__":
    main()
