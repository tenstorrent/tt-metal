# Verification Report: scaled_dot_product_attention

FlashAttention-2 (fused: matmul → online-softmax → matmul, O(S) memory, the
`S_q × S_kv` score matrix never materialized). Phase-0 verified against the
registry model. Verifier CLI is clean on all three loud categories.

## Code Review

Every item below was **fixed in place** (not deferred) unless marked otherwise.

### Fixes applied

1. **Mask `TensorAccessor` rebuilt inside the KV loop (reader).** The mask
   accessor was constructed on *every KV chunk* (`for j … { const auto mask_acc =
   TensorAccessor(…) }`), inside the hot loop, while Q/K/V accessors were built
   once. Hoisted it to function scope alongside q/k/v (`mask_acc` is a function of
   constexpr args + the fixed mask base address — nothing per-chunk). Verified
   both mask and no-mask paths still pass (`test_sdpa_custom_mask`, `test_sdpa_shapes`).

2. **Scale fp32→bf16 conversion used truncation, not round-to-nearest (reader).**
   The resolved attention scale was packed to bf16 via `scale_bits >> 16`
   (truncate-toward-zero), which biases every softmax score slightly low. Replaced
   with the standard round-to-nearest-even (`(scale_bits + 0x7FFF + lsb) >> 16`).
   No overflow risk for the scale value range (small positive float). Exactly
   representable scales (e.g. `1/sqrt(64)=0.125`, `0.25`) are unchanged; non-exact
   scales (`1/sqrt(96)` etc.) round instead of truncate. The dominant residual
   got/true bias (~0.0008) is bf16-intrinsic in the `cb_scores`/`cb_exp`
   intermediates, addressed by the numerical-config refinement — not the scale.

### Reviewed and confirmed correct (no change needed)

- **Blocking-model fidelity — clean.** The three block knobs (`Sq_chunk_t`,
  `Skv_chunk_t`, `KV_DEPTH`) plus `OUT_DEPTH` are computed **once** on the host by
  `_fit_l1(...)` and threaded as compile-time args. Every CB page count and every
  kernel loop bound (`n_kv_chunks`, `n_q_chunks`) is **derived** from them — no
  block dimension is restated as a second literal (DRY holds). **No CB grows with
  a whole-op dimension**: I audited all 17 CBs — `cb_scores`/`cb_exp` are
  `Sq_chunk_t·Skv_chunk_t`, streaming KV CBs are `Skv_chunk_t·Dt·KV_DEPTH`, none
  reference `S_q`/`S_kv`. The load-bearing O(S) constraint is honored. No
  collapsed knob, no half-turned split (the per-core compute loop iterates the
  full `Sq_chunk_t`/`Skv_chunk_t` chunk, not one tile).
- **`_fit_l1` shrink order** matches the design (`Skv_chunk_t → KV_DEPTH(2→1) →
  Sq_chunk_t`) and stops once the working set fits the 1.4 MB budget.
- **CB sync — push == wait/pop for every CB.** Traced the full online-softmax
  recurrence (phases 1–11) across the first-chunk and steady-state branches for
  `cb_row_max`/`cb_row_sum`/`cb_corr`/`cb_out_accum`/`cb_pv`/`cb_exp`/`cb_scores`/
  `cb_q_scaled` — all balanced; resident accumulators are popped exactly once at
  q-chunk end. The `cb_scaler` (1.0, serves both MAX and SUM) and `cb_scale`
  tiles are held (`wait_front`, never popped), pushed once by the reader — correct.
- **Helper usage — every compute phase uses a kernel_lib helper** (`matmul_block`,
  `reduce<MAX/SUM,REDUCE_ROW>`, `mul`/`add`/`sub`/`copy`, `binary_sfpu<BinaryMax>`,
  `unary<Exp>/<Recip>`, `eltwise_chain` fused sub→exp). The running-max is carried
  explicitly with `BinaryMax` (the `MAX+REDUCE_ROW` Accumulate-CB path
  static-asserts, per the design's gotcha) — correctly avoided.
- **API correctness.** `TensorAccessor` (not `InterleavedAddrGen`); `void
  kernel_main()`; includes use `api/dataflow/dataflow_api.h` and
  `api/compute/*`; scaler filled via `calculate_and_prepare_reduce_scaler<…,
  PoolType, ReduceDim>` (confirmed 1.0 for SUM/MAX), never the legacy
  `prepare_reduce_scaler`. Boot uses `compute_kernel_hw_startup + matmul_block_init`.
- **Broadcast dims** match the design's Broadcast Verification table (Scalar for
  pre-scale, Col for the REDUCE_ROW-shaped `m`/`α`/`1/l` broadcasts, None for the
  same-shape adds). `cb_scale` is a one-tile scalar operand; its whole-tile fill
  is a one-time cost (single tile, filled once per kernel, not per work unit) — not
  a redundant per-iteration fill, left as-is.
- **GQA/MQA correctness** is the `kv_head = h/(H/H_kv)` index remap in the reader's
  K/V accessor — no new loop/algorithm, exactly as designed. All mha/gqa/mqa golden
  cells pass.
- **`mcast_pipe.hpp` correctly not used** in phase-1 (reserved for the flash-decode
  and GQA-mcast lamps, neither of which is in the golden TARGET).

### Prompt rules

`eval/prompts/scaled_dot_product_attention.txt` is the older prompt style (no
`## Rules` section), so only the stock native-kernel-support policy applies. Note
its Phase-0 line lists `mask_mode: none, causal`, but `feature_spec.py` exercises
`none, custom` and the op declares `[none, custom]` — the op matches the actual
test contract (the prompt line is a stale typo; `causal` is a refinement).

### Observations (not fixed — noted for later / no lever in scope)

- **`_chunk_size` uses the largest divisor ≤ target, not `min(axis_t,4)` + partial
  remainder** (design's stated approach). This keeps every chunk whole (no
  partial-chunk path needed for phase-0 tile-aligned shapes) and is correct, but
  for a *prime* tile-count `> 4` it would collapse to a 1-tile chunk (granularity
  floor violation). No such shape exists in the current tile-aligned INPUTS
  (worst real case is `Skv_t=6 → chunk 3`), so it is not triggered. The partial-
  chunk path arrives with the non-alignment refinement (R1), which should also
  revisit this so a coarse chunk (4) + a masked remainder replaces the divisor
  trick.
- **Single-head long-context under-fills the grid.** `1×1×2048×64` runs on 16
  cores, `1×1×8192×64` on 64 — because `total_work = B·H·n_q_chunks` is small when
  `B=H=1`. This is exactly the flash-decode (cross-core `S_kv` split) case the
  design flags as a lamp, **but `memory_layout`/sharding is not in the golden
  TARGET**, so there is no golden cell to unlock and no refinement is filed. It is
  also *not* the flagged perf shape (which fills the grid), so it is not a perf
  refinement either. Recorded here only.
- **GQA/MQA KV multicast** (design Lamp 3) would eliminate the per-Q-head DRAM
  re-read of the shared KV head — bandwidth only, correctness already holds. Not
  the flagged perf shape (that is MHA, `H=H_kv=10`), so not filed as a perf
  refinement. Recorded here.

## Registry Conformance

- **INPUT_TAGGERS** — 3 taggers (`tag_alignment`, `tag_attention_kind`,
  `tag_kv_heads`), each with the `(inputs, axes)` signature. ✓
- **SUPPORTED** — every gated axis present: `dtype`, `fp32_dest_acc_en`, `layout`,
  `alignment`, `attention_kind`, `kv_heads_mode`, `mask_mode`, `scale_mode`. Every
  INPUT_TAGGERS key (`alignment`, `attention_kind`, `kv_heads_mode`) is in
  SUPPORTED. ✓
- **EXCLUSIONS** — `[]` (phase-0). ✓
- **validate()** — structural ValueError checks (rank, head_dim, K/V shape, batch,
  GQA ratio, mask dims, causal⊕mask), then SUPPORTED per-axis, then EXCLUSIONS,
  raising the typed `UnsupportedAxisValue`/`ExcludedCell`. Called as the **first
  line** of the public entry point. ✓
- **No `INVALID` in the op file.** ✓ (Confirmed absent.)

### INVALID audit (`eval/golden_tests/.../feature_spec.py`)

`INVALID = []`, and this is **well-formed**: SDPA is TILE-only (no `ROW_MAJOR` in
TARGET), so the canonical `bf8b + ROW_MAJOR` rule is vacuous — there is no such
cell in the cartesian to forbid. No cross-tensor-axis couplings, no
canonicalization cells needed (SDPA is not norm-like with optional weights). Every
`(dtype × everything)` combination is meaningful. The `is_causal ∧ attn_mask`
contradiction is a runtime `ValueError` (op-side), and `causal + cross` is an
op-side EXCLUSION (armed by the causal refinement) — both correctly kept out of
INVALID. No change requested.

### Auto-fixes to SUPPORTED

None — `xpass_drift = 0` (no under-claim). SUPPORTED honestly describes reality.

## Precision Baseline

`test_scaled_dot_product_attention_precision_baseline.py`, bf16 / TILE /
`torch.randn`, default compute config (HiFi4 + fp32 DEST):

| Shape | Max Abs Err | Mean Abs Err | Relative RMS | got/true ratio (med, p5–p95) |
|-------|-------------|--------------|--------------|------------------------------|
| (1,1,32,32)   | 0.00660 | 0.00086 | 0.00474 | 0.99938 (0.977–1.019) |
| (1,1,128,64)  | 0.01006 | 0.00056 | 0.00534 | 0.99935 (0.975–1.026) |
| (2,4,256,64)  | 0.00556 | 0.00040 | 0.00515 | 0.99934 (0.972–1.027) |
| (1,2,1024,64) | 0.00398 | 0.00022 | 0.00569 | 0.99930 (0.968–1.030) |

Golden supported_pass cells (212, bf16+fp32-DEST): **PCC min 0.99996 / median
0.99999 / max 1.0**; normalized rms 0.0026–0.0090.

**Assessment**: excellent for bf16. The got/true ratio is centered on ~1.0 with a
symmetric ±3% spread that widens gently with reduction width — ordinary bf16
rounding noise, **not** a scale/structural bug (which would cluster tightly at a
*non-1.0* constant). PCC degrades only in the 5th decimal as S grows.

**Recommended tolerances** (already codified in the golden `TOLERANCES`): bf16
PCC ≥ 0.995, normalized rms ≤ 0.05; fp32 PCC ≥ 0.999, rms ≤ 0.02.

## Verifier CLI Summary

(`verifier_report.json`, shipped-code run over 2803 golden cases)

- supported_pass: **212**
- xfail_expected: **2473**
- invalid_skipped: 0 (INVALID = [])
- no_axes_found: 118 (regression + translated + loose — outside the registry cartesian)
- **supported_fail: 0** ✓ (must be 0 to ship)
- **xpass_drift: 0** ✓ (must be 0 to ship)
- **xfail_wrong_mode: 0** ✓ (must be 0 to ship)
- supported_marked_xfail: 0

### The 9 `test_regression.py` failures (outside the SUPPORTED cartesian → not a loud category)

All 9 are `severity=precision`, all in `test_regression.py`, on **adversarial
input distributions** (`×10` magnitude, uniform-positive, all-negative) that the
`torch.randn` cartesian never exercises. I triaged them against the scale-vs-precision
rule and **ruled out a scale/structural bug**:

- Probed the got/true ratio on the worst cell (`test_negative B1_H12_S512`, PCC
  0.987, normalized rms 0.168): **ratio median 0.99925, std 0.0018** — the output
  is correct to ~0.1% per element, centered on 1.0.
- The `rms` metric is **normalized by the reference stddev** (`helpers.py:156`).
  Uniform/negative inputs drive attention to near-uniform, so the reference output
  is nearly constant (e.g. mean −0.998, std 0.0125); a ~2-ULP bf16 error
  (`max_abs` 0.0078) divided by that tiny stddev inflates to rms 0.16. The failure
  is a **metric artifact on a degenerate reference**, plus genuine bf16 precision
  on peaked softmax (`test_large_magnitude`).

These are not a kernel bug and not blocking (verifier categories clean). They are
the natural target of the **numerical-config refinement (R2)** — float32 dtype +
fp32 intermediate CBs push these back under tolerance. Cross-referenced there.

## Recommendations

- **Refinement priority / ordering** is set in `op_requirements.md`. The one
  non-obvious constraint: the perf-flagged loose case (`1×10×9472×128`) pins
  `fp32_dest_acc_en=False`, which is **not** in phase-0 SUPPORTED, so that shape is
  currently `xfail`. The first perf refinement (R3) therefore **depends on** the
  numerical-config refinement (R2) adding `fp32_dest_acc_en=False` — R2 is pulled
  ahead of the (harder) causal refinement in the queue for exactly this reason.
- **Perf headroom exists** on the flagged shape: the reader uses a
  read-one-tile / barrier / push pattern (the `double_buffer` anti-pattern) even
  though the KV CBs are double-buffered, so the shape is latency-bound on DRAM
  reads. Largest measured supported cell (`1×8×4096×128`, 3.47 ms / 110 cores)
  extrapolates to ~5× that for the flagged shape vs a 0.35-math-util (~2.16 ms)
  target — ample room. This is R3.
- **Memory pressure**: `_fit_l1` bounds the working set to 1.4 MB and shrinks the
  KV chunk for large `D` (e.g. `D=1024 → Dt=32`); no OOM observed across the golden
  suite including `S=8192` and `D=1024`. No memory refinement needed for phase-0
  shapes.
- **bf16 intermediate precision** (`cb_scores`/`cb_exp`) is the residual precision
  ceiling; it has a concrete lever (fp32 intermediate CBs + `UnpackToDestFp32`)
  and is folded into R2 (`/numeric-formats-metal`), not filed separately.
