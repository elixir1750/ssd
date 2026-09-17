# DFlash for every possible acceptance position

This experiment prepares the next DFlash proposal for **every `k = 0..K`**
before the target verifies the current `K` candidates. It keeps the true bonus
and does not guess or enumerate it. The default implementation is a **single-GPU serial execution reference**.
With `--draft-async`, SSD runs the target on GPU 0 and all-position drafting on
GPU 1, overlapping branch preparation with current target verification.
Real-model NCCL correctness and speedup still require GPU validation.

The native target attention, argmax rules, verification and numeric reference
thresholds are unchanged. The previously observed native-target AR/block kernel
differences remain unresolved and must not be treated as a passed GPU check.

## Server command

From the repository root, using the existing SSD environment and environment
variables described in [dflash.md](dflash.md):

```bash
python -O bench/dflash.py \
  --target /path/to/Qwen3-target \
  --draft /path/to/matching-original-DFlash \
  --block-size 8 --max-new-tokens 128 \
  --temperature 0 --all-positions \
  --output results/dflash-all-positions.json
```

`block-size=8` means **one known anchor and K=7 candidate tokens**. Therefore
there are up to **8 acceptance-position branches**, k=0 through k=7. To test
eight new candidates, use `--block-size 9`. Run the same command without
`--all-positions` to obtain the ordinary synchronous DFlash baseline. Temperature
sampling and separate draft temperature use the same exact-rejection routine.

Python API: add `dflash_all_positions=True` alongside `use_dflash=True`.
The serial mode requires one GPU; the async mode requires exactly two GPUs.
Both modes require one active request and eager execution.

## A round, with positions made explicit

At round start, `C` is the committed prefix of length `L`; `a` is a known bonus
from the previous round, at position L. Current candidates are `x1..xK`.

```text
Current target input:       [a, x1, x2, ..., xK]

Hypothetical accepted k:    C a x1 ... xk
Future committed length:   S = L + 1 + k
Future bonus position:     S (unknown before current verification)

Branch draft query:        [last-known-token, MASK-bonus, MASK-y1, ...]
Query positions:            S-1              S           S+1
Saved proposal:                                         y1, y2, ... and q

After real verification:   actual k, actual bonus z
Next target input:         [z, y1, y2, ...]
Next acceptance tests:         y1, y2, ... using their saved q
```

For k=0, the clean last-known token is `a`; for k>0 it is `xk`. The first MASK
slot exists in the draft forward but is **not projected or sampled as a bonus**.
Only hidden states after that slot become saved candidate probabilities/tokens.
Changing the real bonus does not regenerate the proposal or change its q.

## Which KV is retained?

There are two separate states:

1. **Canonical context KV** comes only from verified target features, projected
   through DFlash's original fusion, K/V projections, key norm and RoPE.
2. **One-generation draft self-KV** is saved from the current proposal forward.
   It contains anchor/MASK representations, not a re-encoding of sampled clean
   tokens. This is deliberately an approximate context for the next draft.

For branch k, combine canonical context before L with k proxy positions
`L..L+k-1`, then re-encode the clean last-known token at L+k. Excluding this last
position from the proxy avoids duplicating its KV. The branch query then contains
the unknown-bonus slot and future candidate slots. Branches do not read each
other's KV or mutate canonical context. No proxy KV ever enters the target.

After verification, discard unselected branches. Keep only the selected tokens,
their actual q, and their own-block KV (unknown bonus plus candidates). At the
next round, append the newly accepted target features to canonical context;
this replaces temporary history for **new branch construction**. It does not
retroactively recompute the selected proposal or its saved self-KV.

Each branch is a separate ordinary bidirectional DFlash forward. Branch sampling
uses a persistent independent generator, so preparing future proposals does not
consume the randomness used by current target verification. Position selection
depends only on the current acceptance result, never future candidate quality.

This construction keeps exact proposal accounting for the original DFlash
independent-position sampler, but does not promise the same acceptance length
as ordinary DFlash: current target features and the real future bonus were not
available when those candidates were generated. DFlash2 is not supported.

## Boundaries and measurements

All positions are prepared if a continuation can use candidate tokens. Branches
that would end at EOS or have at most one output/context slot remaining are
skipped; that final single slot needs only the true bonus. Candidate lengths are
shortened near output/context limits. Initial generation uses an ordinary
DFlash proposal to bootstrap; subsequent nonterminal rounds use selected saved
proposals. Stale sequence/position/length tags raise errors rather than silently
using an unrelated proposal.

The JSON report contains:

- `mode: dflash_all_positions_serial`;
- `metrics.dflash_position_rounds`: current length, source acceptance position,
  prepared branch lengths, skipped positions, preparation time and accepted count;
- `position_summary`: acceptance grouped by the **previous round's k** that
  produced the consumed proposal, plus a separate bootstrap group.

`mean_accepted` excludes the known anchor. These are measurements of selected
proposals under the native target, not counterfactual acceptance of every branch
or proof of equivalence to the HF reference. Serial preparation time is actual overhead and must not be reported as hidden
latency. Async rounds additionally record `branch_wait_ms`, the target's host
wait for draft completion after verification. This is not itself a GPU overlap
measurement or a speedup claim.

There are up to K+1 draft forwards per preparation and O(K²) saved candidate
positions. Full FP32 q storage costs O(K² × vocabulary size); start with a small
block. Historical context is not kept as a full independent copy for every saved
branch; only own-block KV survives each forward. The GPU memory reservation
includes extra branch work space, retained self-KV and q tensors.

## CPU validation and remaining GPU work

```bash
python -m unittest discover -s tests -p 'test_dflash*cpu.py' -v
```

The tests cover all k including endpoints; clean anchor and unknown-bonus
position alignment; canonical-KV projection parity; RNG isolation; frozen q
after bonus/feature arrival; stale proposal rejection; EOS/context/output limits;
and repeated end-to-end rounds with a tiny real draft and deterministic target
fixture. Baseline tests and the optional pinned-upstream forward check also run.

Real-model acceptance, native CUDA/NCCL behavior and memory requirements remain
to be measured on the server. CPU/Gloo tests exercise the actual spawned worker,
real tiny DFlash and HF target, all selected positions, frozen FP32 q transport,
multiple requests and terminal slots, stale messages, worker errors and timeouts.
A gated worker test verifies that target work can proceed while preparation is
outstanding; this does not establish CUDA kernel overlap or GPU performance.

## Two-GPU SSD execution

```bash
python -O bench/dflash.py \
  --target /path/to/Qwen3-target \
  --draft /path/to/matching-original-DFlash \
  --all-positions --draft-async --block-size 8 \
  --max-new-tokens 128 --temperature 0 \
  --output results/dflash-two-gpu.json
```

API: `LLM(target, draft=draft, use_dflash=True, dflash_all_positions=True,
 draft_async=True, num_gpus=2, max_num_seqs=1, speculate_k=7)`.
Launch the script normally: `LLMEngine` spawns rank 1, as in SSD's existing
async engine. Do not launch two copies with `torchrun`.

Rank 0 reuses `ModelRunner.run_dflash_target`, the target paged KV pool,
`DFlashScheduler`, `DFlashStep` and exact p/q rejection sampling. Rank 1 runs
`DFlashPositionsRunner`. It receives immutable copies of target embedding and
LM-head weights once during initialization (tied weights share one allocation).
It owns canonical draft context and all proxy branches. The full target model
is not loaded on rank 1.

Both ranks use SSD's target TP / target-draft NCCL group layout and tensor
transport. A local pipe carries command IDs, request metadata, completion and
errors; candidates, FP32 q and accepted features travel directly over NCCL.
The per-engine file rendezvous is local, avoiding collisions on a fixed TCP port.
No target feature from the current verification enters preparation of its next
branches. Only the selected branch survives, with its original q.

Round order:

1. Receive current candidates/q from rank 1.
2. Start rank-1 branch preparation without waiting for completion.
3. Verify the current block and run rejection sampling on rank 0.
4. Wait for remaining preparation, select the accepted-position branch and send
   only accepted target features to rank 1.
5. Refresh canonical context before consuming the saved next proposal.

The first proposal is bootstrapped; the final anchor-only step bypasses drafting.
Both EOS and output/context limits reset remote state before another request.
Call `llm.exit(hard=False)` in a `finally` block: shutdown drains pending work,
reaps the owned child and closes process groups. Worker errors and stale protocol
messages fail the request; `distributed_timeout_seconds` bounds control and
process-group waits (default 180 seconds). Failure does not silently fall back
to a different sampler.

All acceptance-position forwards still execute sequentially **on the draft
card**. Preparing K+1 branches may exceed target verification time; the target
then waits. End-to-end performance must include weight/features/q transfers,
branch preparation and waiting. Compare against both AR and serial FDFlash.
The known native single-token/block numerical differences remain unchanged.

GPU regression entry point:

```bash
python bench/verify_dflash_dual.py --target /path/to/Qwen3-target \
  --draft /path/to/matching-original-DFlash --block-size 8 \
  --output results/fdflash-dual.json
```

It runs ordinary single-GPU DFlash, serial FDFlash and async FDFlash in separate
processes, checks five serial/async greedy
outputs for exact equality, and exercises two positive-temperature settings,
request resets, page crossing, one-token output and the context limit. Sampling
cases test execution only. All results are persisted and failures exit nonzero;
existing output files are not overwritten. Performance uses three fixed prompts,
128 output tokens (`ignore_eos=True`), one full-length warmup per prompt and
three measured repeats. Request wall time includes prefill, transfers, drafting,
verification, sampling and reset. Initialization/warmups are reported separately.
Aggregate throughput uses total tokens divided by the sum of per-prompt median
latencies; output differences between modes are reported explicitly. A passed comparison would establish
serial/async parity for those cases, not native AR or HF equivalence.

Local validation for this implementation: 23 CPU tests passed, including the
optional pinned-upstream parity test. The GPU result below covers the two-GPU path.

## A800 validation and throughput (2026-09-17)

Job `ssd-fdflash-dual-0917-1` ran commit `d70f111` with Qwen3-8B and
Qwen3-8B-DFlash-b16 on two A800 80GB GPUs, block size 8, one request and a
512-token context. All 45 functional requests completed; all five serial/async
greedy comparisons passed and the draft worker exited cleanly.

| Mode | GPUs used | Aggregate generated token/s |
| --- | ---: | ---: |
| Ordinary DFlash | 1 | 70.09 |
| Serial all-position FDFlash | 1 | 24.99 |
| Async all-position FDFlash | 2 | 49.40 |

Three fixed prompts each generated exactly 128 tokens with greedy sampling and
`ignore_eos=True`. Each prompt had one full-length warmup and three timed repeats.
Aggregate throughput is total tokens divided by the sum of median request
latencies, including prefill, transfers, drafting, verification, sampling and
reset, but excluding initialization and explicit warmups. All three modes had
identical measured output tokens. This is a small workload comparison, not a
broad serving benchmark or native AR/HF equivalence claim.

Async FDFlash was 1.98x serial FDFlash, but only 0.705x ordinary single-GPU
DFlash. Mean committed tokens per round fell from 2.10/3.20/8.00 in ordinary
DFlash to 1.51/2.37/5.33 in both FDFlash modes. Async branch preparation still
left 0.63/0.41/0.19 seconds of residual wait per request. Acceptance quality
and branch preparation cost remain the main optimization targets.
