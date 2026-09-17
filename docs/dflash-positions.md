# DFlash for every possible acceptance position

This experiment prepares the next DFlash proposal for **every `k = 0..K`**
before the target verifies the current `K` candidates. It keeps the true bonus
and does not guess or enumerate it. The implementation is a **single-GPU serial
execution reference**, not a two-GPU asynchronous speedup implementation. Both
the branch forwards and the target forward run sequentially in this version.

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
The mode still requires one GPU, one active request and eager execution.

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
or proof of equivalence to the HF reference. Serial preparation time is actual
overhead here and must not be reported as hidden latency or an SSD speedup.

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

Real-model acceptance, native CUDA behavior and memory requirements remain to be
measured on the server. A later implementation can overlap this preparation
with target verification on a separate device after this data flow is validated.
