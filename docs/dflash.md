# Standard DFlash baseline in SSD

For the optional next-stage experiment, see [DFlash for every acceptance
position](dflash-positions.md), enabled with `--all-positions`. The default mode
described below remains the standard synchronous baseline.

This stage connects **original DFlash** to SSD's native dense Qwen3 target.
It performs the normal synchronous cycle: target features → one DFlash forward
→ target verification → accepted target features. There is no SSD lookahead
tree, unknown-bonus slot, cross-round MASK-KV retention, or asynchronous work.

## Supported scope

- One GPU, one active request, eager execution. Multiple prompts run serially.
- A matching original DFlash checkpoint and dense Qwen3 target, in local folders.
- Greedy decoding and temperature sampling, including separate draft temperature.
- Target paged KV; draft uses its own `DynamicCache` and discards anchor/MASK KV.
- EOS, maximum generation length, and a shortened final block at the context limit.
- Prefix-cache reuse is disabled because it would omit required target features.

Batching, tensor parallelism, CUDA graphs, DFlash2, sliding-window models, top-p/
top-k sampling and SSD asynchronous mode are not implemented in this stage.
Unsupported mode combinations fail at configuration time. Performance from this
baseline is not comparable to a fully optimized DFlash serving implementation.

## Run on the CUDA server

Use the repository's existing environment (`uv sync`; Python 3.11–3.12,
Torch 2.8.0 / Transformers 4.57.1). No separate DFlash package installation is
needed. A small MIT-licensed copy of the original draft architecture is included;
installing the latest upstream package would introduce unrelated DFlash2 and
different dependency versions.

From the repository root:

```bash
# SSD currently requires these at package import, even for explicit local paths.
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=9.0  # set for your GPU, e.g. 9.0 for H100

python -O bench/dflash.py \
  --target /path/to/Qwen3-target \
  --draft /path/to/matching-original-DFlash \
  --block-size 16 --max-new-tokens 128 \
  --temperature 0 --check-reference \
  --output results/dflash-greedy.json
```

`block-size=16` means one known anchor and **15 proposed tokens**. Use `--chat`
to apply the tokenizer's chat template. The default prompt is raw text.

`--check-reference` first loads HF Qwen3, saves selected-layer prefill features,
last-position logits and greedy output, then releases that model before loading
SSD. It checks feature/logit agreement with explicit BF16 tolerances and exact
greedy token equality. Numerical kernel differences can change a near-tied
argmax; a mismatch is reported as a failed check, never silently accepted.
This checks the native target interface and lossless greedy output; it does not
claim pretrained DFlash acceptance parity or a stochastic sequence match.

Then exercise the sampling path:

```bash
python -O bench/dflash.py \
  --target /path/to/Qwen3-target \
  --draft /path/to/matching-original-DFlash \
  --block-size 16 --temperature 0.8 \
  --output results/dflash-sampling.json
```

Python API:

```python
from ssd import LLM, SamplingParams

llm = LLM(target_path, draft=draft_path, use_dflash=True,
          speculate_k=15, num_gpus=1, max_num_seqs=1)
outputs, metrics = llm.generate(["Explain speculative decoding."],
                                SamplingParams(temperature=0, max_new_tokens=128))
llm.exit(hard=False)
```

## State and numerical invariants

At round start, the target cache covers the committed prefix, while the next
known anchor is held in `recovery_token_id`. Draft context comprises its previous
target-derived KV plus a pending delta of newly accepted target features. The
delta includes the **old anchor and accepted candidates**, not the new bonus or
the rejected tail. After drafting, crop draft cache to the committed prefix.
Verify `[anchor, candidates...]`; commit `[anchor, accepted candidates...]` and
hold the newly sampled bonus as the next anchor. Keep target cache logically
truncated at the committed length; rejected slots are overwritten next time.

SSD's Qwen3 layers defer residual addition. Feature extraction reconstructs that
sum, preserves checkpoint layer order, and uses final-normalized output if the
checkpoint selects the last layer (matching HF `hidden_states[layer_id + 1]`).
Draft embedding and output weights are shared with the native target in this
single-device implementation.

Sampling uses float32 probabilities from the actual independent per-position
draft proposal. It always applies p/q acceptance and positive-part residual
correction; it does not use SSD's asynchronous cache-miss verification shortcut.
An argmax draft is a point-mass proposal. No bonus is guessed in this stage:
the normal, already-known bonus is the draft input anchor.

## Validation

```bash
python -m unittest discover -s tests -p test_dflash_cpu.py -v
```

Optionally set `DFLASH_REFERENCE_MODEL=/path/to/pinned-dflash/dflash/model.py`
to compare three draft forwards against the pinned upstream architecture using
identical tiny random weights. This does not download pretrained weights.

The CPU suite runs actual tiny DFlash attention/cache operations and tests
residual accounting, rejection sampling, accepted-feature slicing, serial
state updates, EOS, final blocks and request reset. It bypasses only SSD's
CUDA-only package initializer. Native target CUDA kernels, real checkpoint
loading on GPU, acceptance rates and throughput still require the server check
above. This local integration must not be described as GPU-validated yet.

The architecture was adapted from z-lab/dflash commit
`07ebd93db9f472af339b644bb70221ad8428328a`; see `third_party/DFLASH_LICENSE`.
