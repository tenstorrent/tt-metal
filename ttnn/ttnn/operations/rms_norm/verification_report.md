# Verification Report: rms_norm

## Code Review
- **Fixed**: the entry point had no `memory_config` kwarg. The golden harness passes `memory_config=` for every sharded cell, so 4786 cells died with `TypeError` (`xfail_wrong_mode`, category `signature`). Added `memory_config: Optional[ttnn.MemoryConfig] = None` to `rms_norm()` and `validate()`. A non-SUPPORTED output `memory_layout` raises `UnsupportedAxisValue`. An interleaved `memory_config` is honored for output allocation (default DRAM).
- Design conformance: algorithm (Σx² → fp32 inv_rms in place → scale [× gamma]), RISC ownership, and the R1/R2 regime split match `op_design.md`. Grid is full (`split_work_to_cores`, row_wise=True). Both reader and writer move whole segments (Rb×Sb tiles) per barrier. CBs are double-buffered.
- Blocking-model fidelity: all block knobs (`Rb`, `Wb`, `Sb`, `x_depth`, `gamma_resident`) come from `pick_blocking()` and every CB size / CT arg derives from it (single source, DRY). `cb_x` scales with `Wt` only under the predicate-guarded resident fast path (`Wt*tile ≤ X_BUDGET`), with a streamed R2 fallback — sanctioned, not a collapsed knob. No finding.
- Observation (not fixed, folded into perf refinement): `Sb` is the largest divisor of `Wt` ≤ 16, so prime/awkward `Wt` (e.g. 37, 53, 89, 127) gets `Sb=1`. That is a half-turned granule — it collapses per-segment amortization for those widths. Needs a ragged-tail segment, which is a kernel change. Filed in the perf refinement.
- Observation: gamma is scattered tile-by-tile by the reader with scalar L1 loops; fine for resident gamma (once/core) but costs per row block in the streamed-gamma path.
- Helpers: compute uses `eltwise_chain`, `reduce` with `Accumulate`, `tilize`, `untilize`; the reader uses `prepare_reduce_scaler`. The raw dataflow (stick-segment reads with zero-fill, gamma scatter) has no helper. No mcast is used in Phase 0.
- Includes / kernel syntax / TensorAccessor: OK.

## Registry Conformance
- Confirmed: `INPUT_TAGGERS` (signature `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS` (empty), `validate()` present; `validate()` is the first call of the entry point and checks SUPPORTED then EXCLUSIONS. Op file does not declare `INVALID`.
- No drift auto-fixes needed (`xpass_drift` = 0).
- INVALID audit (`feature_spec.py`): bf8b+RM for activation and gamma present; no-gamma canonicalization present; no cross-tensor couplings except the author-scoped "for now" entries (RM input × sharded × TILE gamma), which couple input layout with gamma layout under sharding. These are documented in-file as parked for now, not structural impossibilities. Noted as an advisory: they hide refinement signal, and should be removed once sharded support exists.
- Advisory: 2 `invalid_unexpected` cells (`test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[*-bfloat8_b]`) are xfail in the translated tests but not skipped by INVALID. They are not counted as failures.

## L1 Ledger Audit
- Ledger currency: ledger lists `cb_accum` as "removed" but its CB-index column in `op_design.md` still numbers cb_accum/cb_stat differently from the code (`CB_STAT=5`, `CB_XN=6`). The sizes/expressions match the code. `cb_xn`, `cb_gamma_rm` rows exist. OK.
- Size vs live set: `cb_x` x_depth=2 over live Rb*Wb is the stated prefetch gap. No unexplained over-capacity.
- Page format vs DEST: intermediates are Float32 and `fp32_dest_acc_en=True` — consistent (no Float32-page-on-16-bit-DEST finding). When `fp32_dest_acc_en=False` is added (Refinement 1), `cb_sq`/`cb_xn`/`cb_stat` should drop to bf16 (see refinement).
- Disjoint lifetimes: `cb_sq` (pass 1) vs `cb_out_tiles` (pass 2) alias candidate is justified in the ledger (≤64 KB). OK.
- Block-size defaults held: interleaved splits rows across the full grid then takes `Rb*Sb ≤ 16`. Departure `Rb=1` for Wb>16 is justified in the design. No measurement of `G_CAP` yet — folded into the perf refinement.
- Footprint: `x_depth*Rb*Wb*Tx + [RM] 64*Sb*32*es + gamma(Wt|2Sb)*Tg + G*T32(sq)+G*T32(xn) + Rb*T32 + 2G*Tx + [RM] 2Sb*Tx`; `cb_x` scales with `Wb` (≤ X_BUDGET), `cb_gamma` with `Wt` (≤ GAMMA_BUDGET). Everything else scales with `G ≤ 16`.
- Traffic budget present and consistent with the implementation (R1: x 1×, R2: x 2×; R3 W-split deferred with a positive reason). Carried into Refinement 2.

## Precision Baseline (test_rms_norm_precision_baseline.py; TILE input, RM gamma, randn)
| Shape | dtype | PCC | Max Abs | Mean Abs | Rel RMS | got/true p5 / median / p95 |
|-------|-------|-----|---------|----------|---------|----------------------------|
| (1,1,32,64) | bf16 | 0.9999985 | 0.0155 | 9.5e-4 | 1.8e-3 | 0.9964 / 0.9992 / 1.0019 |
| (2,1,100,47) | bf16 | 0.9999985 | 0.0206 | 1.1e-3 | 1.9e-3 | 0.9964 / 0.9992 / 1.0019 |
| (1,1,128,4096) | bf16 | 0.9999985 | 0.0393 | 9.5e-4 | 1.8e-3 | 0.9970 / 0.9998 / 1.0027 |
| (1,1,32,8192) | bf16 | 0.9999984 | 0.0385 | 9.7e-4 | 1.8e-3 | 0.9976 / 1.0005 / 1.0033 |
| (1,1,32,64) | fp32 | 0.9999997 | 0.0135 | 8.1e-4 | 1.5e-3 | 0.9974 / 0.9987 / 0.9998 |
| (1,1,128,4096) | fp32 | 0.9999997 | 0.0212 | 5.1e-4 | 9.8e-4 | 0.9981 / 0.9994 / 1.0006 |
| (1,1,32,8192) | fp32 | 0.9999997 | 0.0188 | 3.9e-4 | 7.5e-4 | 0.9988 / 1.0000 / 1.0013 |

**Assessment**: ratio clusters at 1.0 with a narrow spread — rounding noise, no scale bug. fp32 rel-RMS (~1e-3) is dominated by the non-fp32 path through FPU (SrcA/B TF32-style unpack for the scale multiply); acceptable.
**Recommended tolerances**: PCC ≥ 0.999 (fp32) / 0.995 (bf16); the HiFi2 perf cases need 0.9995.

## Verifier CLI Summary (golden, 40836 cells)
- supported_pass: 436
- xfail_expected: 6481
- invalid_skipped: 33900
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0 (was 4786 before the `memory_config` signature fix)
- invalid_unexpected: 2 (see above), infeasible_skipped: 2, no_axes_found: 15 (regression tests, all passed)
- The existing acceptance tests pass (111 tests).

## Recommendations
- The `xfail_expected` bucket is the refinement queue (bf8b, `fp32_dest_acc_en=False`, TILE/bf8b gamma, sharded memory layouts). Covered in `op_requirements.md`.
- fp32 input with `fp32_dest_acc_en=False` stays in EXCLUSIONS (precision convention).
- The wide LOOSE_CASES (`1x1x32x16384`, `32x32768`, `64x12288`) and decode shapes under-fill the grid (R3 W-split) — Refinement 2.
