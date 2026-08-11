# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-08-11
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Multi-core from day 1: the grid is partitioned into row-groups; the independent `row`
  axis is split across groups and the **dependent** `hidden` axis is split *within* a group, with
  partial `Σ x²` combined over the NoC (gather-to-root + `mcast_pipe` multicast-back). Both regimes
  ship (R1 `w_group_size == 1` row-parallel, R2 `w_group_size > 1` cross-core). Native TILE and
  ROW_MAJOR I/O (in-kernel tilize/untilize, no host-side `to_layout` / `pad` / `slice`), and the
  ragged hidden tile is masked numerically before squaring so the RMS denominator counts only valid
  elements.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED].
  EXCLUSIONS=[{dtype: float32, fp32_dest_acc_en: False}].
- **Accuracy achieved** (measured on 5 shapes × 2 dtypes × 2 layouts × gamma/no-gamma via
  `test_rms_norm_precision_baseline.py`; TILE + gamma rows quoted):
  bfloat16 — PCC = 0.999997, max_abs ≤ 0.066, mean_abs ≈ 0.0012, rel_rms ≈ 0.0023–0.0026,
  got/true ratio median 0.9999–1.0002 (p5/p95 ≈ ±0.4 %);
  float32 — PCC = 0.9999997, max_abs ≤ 0.027, mean_abs ≈ 0.0007, rel_rms ≈ 0.0012–0.0015,
  ratio median 0.9989–0.9991 (p5/p95 ≈ ±0.13 %; a systematic ~0.1 % shrink from the FPU's
  truncating fp32 mantissa, not a scale bug).
- **Golden suite at Phase 0**: 737 / 737 supported cells passing (`supported_pass` = 737,
  `xfail_expected` = 6174, `invalid_skipped` = 33900, and **0** in each of `supported_fail`,
  `xpass_drift`, `xfail_wrong_mode`), per `verifier_report.json`.
- **Issues encountered**: no drift and no functional failures — the SUPPORTED block was already
  honest. Verifier code-review fixes: (1) the CB inventory was stated **twice** (the descriptor's CB
  list and the L1-residency solve's `fixed_bytes` / `per_row_bytes` expressions), so a page-count
  change could silently drift the solve — the inventory is now one `_cb_specs()` table and the solve
  *derives* the footprint from it by differencing (verified byte-identical to the previous closed
  form over 4608 configurations); (2) `cb_input_tiles` reserved a depth-2 prefetch slot even when the
  busiest core's whole row assignment is a single block (e.g. `(1,1,8192,1024)`), where no block
  `b+1` exists — the second buffer is now not allocated in that case; (3) the reader / writer read
  their `TensorAccessorArgs` / `McastArgs` from hardcoded arg offsets that no assertion pinned, so an
  added host arg would have silently fed a peer's args — the three offsets are now named host
  constants with build-time asserts. `l1_ledger.md` updated for all of the above plus a stale
  measured-selection row.
- **Tests added**: `test_rms_norm.py` (acceptance, 82 cases — immutable spec),
  `test_rms_norm_shapes.py` (adversarial-shape sweep, 92 cases),
  `test_rms_norm_perf.py` (device-ns harness + the P1/P2 knob sweeps),
  `test_rms_norm_precision_baseline.py` (**new** — PCC / abs / rel-RMS / got-true ratio spread,
  40 cells).

## Refinement 1 — Numerical configurability expansion (`fp32_dest_acc_en=False` + `bfloat8_b`)
- **Date**: 2026-08-11
- **What was done**: extended SUPPORTED to `fp32_dest_acc_en=[True, False]`,
  `dtype += bfloat8_b`, `gamma_dtype += bfloat8_b`, keeping
  `{dtype: float32, fp32_dest_acc_en: False}` in EXCLUSIONS (now REACHABLE and actively enforced
  rather than a declared-but-dead entry). **Zero kernel changes** — all three `.cpp` files are
  byte-identical to Phase 0; `/numeric-formats-metal`'s pass condition held, so the entire diff is
  the op file + the program descriptor + tests.
  Reused rather than rebuilt: (1) DEST capacity is only ever read through `DEST_AUTO_LIMIT` /
  `Dst::D0`, so the 4 → 8 doubling when fp32 DEST accumulation is off needed nothing — no literal
  in the kernels or the L1 solve assumed the halved value; (2) the whole `cb_stat_*` chain
  (`cb_stat_sq`, `_partial`, `_gather`, `_sum`, `cb_rstd_send`, `cb_rstd`) plus `cb_zero_tile` were
  already pinned to fp32 **unconditionally** instead of following `fp32_dest_acc_en`, and
  `cb_scaler` / `cb_wmask` already stay bf16 (`reduce_helpers_dataflow.inl:185-187` `static_assert`s
  the scaler format) — so `Σ x²` still crosses L1 in fp32 and only the *in-DEST* accumulation
  narrows, which is what keeps `row_reduce_accumulate`'s failure mode away; (3) `_cb_specs()`
  already derived every input/output/gamma CB format from the corresponding tensor's dtype, so
  `bfloat8_b` flowed through as a third `tile_size` value (1088 B) with no new branch — and because
  a bfp8 tile is *smaller* than bf16/fp32, the residency predicate only gets easier and the selected
  `(G, C, R)` can only get coarser.
  The one real code change: `_elem_bytes()` / `_BLOCK_FLOAT_DTYPES` in the descriptor, because
  `Tensor.element_size()` **raises** for block-float ("datum for bfp2, bfp4, bfp8 is invalid"). It
  returns 0 for `bfpN` — the only consumers of an element size are the ROW_MAJOR stick legs, and a
  block-float tensor has no ROW_MAJOR form (ttnn itself `TT_FATAL`s `layout == Layout::TILE`), which
  independently confirms `feature_spec.INVALID` is right to *skip* those cells rather than xfail them.
- **Accuracy achieved** (vs the golden suite's own per-dtype `TOLERANCES`, on shapes
  `(1,1,32,64)`, `(1,1,64,128)`, `(4,8,32,256)`, `(1,1,32,4096)`, `(1,1,32,8192)`, `(2,4,128,512)`,
  `(1024,1024)`, plus the cross-core-forcing `(1,1,32,16384)`, `(1,1,32,32768)`, `(1,1,64,12288)`,
  `(1,1,160,11008)`):
  bfloat16 @ `fp32_dest_acc_en=False` — PCC ≥ 0.999937, rel-RMS ≤ 0.012 (gate 0.995 / 0.04);
  bfloat8_b (both DEST settings, incl. the wide cross-core combine) — PCC ≥ 0.999854,
  rel-RMS ≤ 0.021 (gate 0.99 / 0.10);
  `gamma_dtype=bfloat8_b` against bfloat16 and float32 activations — PCC ≥ 0.999937,
  rel-RMS ≤ 0.0117;
  `pad_poison` (poison 1000.0) @ `fp32_dest_acc_en=False` + HiFi2 — PCC ≥ 0.999979 on all 6 shapes,
  i.e. the mask-before-square identity `(x·mask)² == x²·mask` holds under a 16-bit DEST accumulator.
  rtol/atol: the precision matrix reports max/mean-abs and the got/true ratio spread per cell rather
  than gating on a single rtol/atol pair; the scale tripwire is `|ratio_median − 1| < 0.02`
  (0.05 for bfloat8_b, 0.08 at LoFi — see Issues).
- **Golden test progress**: targeted slices, all clean (the harness re-runs the full suite).
  `1x1x64x128 × INTERLEAVED` = **48 passed / 12 xfailed** — and 48 is exactly what SUPPORTED now
  predicts (bf16 × 2 DEST settings × 6 gamma combos × 2 layouts = 24, fp32 × 1 setting = 12,
  bfp8 × 2 settings × 6 gamma combos at TILE only = 12), with the 12 xfails being precisely the
  `{float32, fp32_dest_acc_en=False}` EXCLUSIONS cells → **no XPASS drift**. Non-aligned shapes
  `1x1x32x50` + `1x1x17x64` × INTERLEAVED = 72 passed / 24 xfailed. `-m pad_poison` = 6 passed
  (all interleaved) / 18 xfailed (the 3 sharded placements — Refinement 2). `-m perf` = **8 passed**
  (every interleaved perf case, now running at its pinned `fp32_dest_acc_en=False` + HiFi2) /
  5 xfailed (sharded — Refinements 2/5). This is the gate Refinements 3 and 4 were waiting on:
  they can now measure their specified configuration instead of proxying it at
  `fp32_dest_acc_en=True`.
- **Issues encountered**: two, neither an op defect.
  (1) **LoFi systematic shrink.** `test_rms_norm_precision_matrix_gamma_dtype` initially failed 2 of
  140 cells (LoFi + bf16) on the got/true-ratio tripwire at ratio median 0.965 — a ~3.5 % *uniform*
  shrink. Cause is hardware, not the kernel: LoFi truncates srcA/srcB to a 5-bit mantissa and the
  FPU **truncates rather than rounds**, so each of the `x · rstd` and `· gamma` multiplies biases
  low and the two compound. PCC stayed ≥ 0.9995 throughout. Same mechanism Phase 0 already recorded
  as a ~0.1 % shrink at fp32/HiFi4. Fixed in the *test*: the scale-bug band is now derived from
  `math_fidelity` and dtype (0.02 base → 0.05 for bfloat8_b → 0.08 at LoFi) so the tripwire keeps
  detecting real scale bugs instead of firing on documented hardware behavior.
  (2) **`element_size()` raises for block-float** — see above. Found on the first bfp8 probe, before
  any kernel work, which is why bfloat8_b needed no kernel changes at all.
  Nothing was added to EXCLUSIONS: every cell this refinement named measured green.
- **Tests added**: `test_rms_norm_precision_matrix` + `test_rms_norm_precision_matrix_gamma_dtype`
  in `test_rms_norm_precision_baseline.py` (140 cells: 5 shapes × 3 dtypes × 2 `fp32_dest_acc_en`
  × {HiFi4, HiFi2} × {normal, uniform}, plus an independent gamma_dtype × {HiFi4, LoFi} sweep;
  uniform inputs are kept deliberately — all-positive data is the monotonically-growing-sum regime a
  narrowed DEST accumulator degrades on). Skips are only the two declared-impossible sets
  (`{float32, accFalse}` = EXCLUSIONS, `bfloat8_b × non-tile-aligned` = `feature_spec.INVALID`).
  `test_rms_norm_shapes.py` gained an `fp32_dest_acc_en` axis (92 → 184 cases): its whole premise was
  "the golden resilience / pad_poison cases pin `fp32_dest_acc_en=False`, which is unsupported, so
  replay them at `True`" — that premise is now obsolete, so it sweeps both DEST datapaths and its
  docstring says why. Unit directory: **464 passed / 36 skipped / 0 failed** (was 372/36/0).
