# Changelog: div

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  `ttnn.ProgramSpec` on Quasar, one `ttnn.generic_op` per call; thread plan `THREAD_PLAN = (4, 4, 2)`,
  `BLOCK_TILES = 4`, `BUFFER_DEPTH = 2` (one host constant each; kernels take the counts as CT args).
  Whole rounds of C tiles over the full grid (`split_work_to_cores(..., row_wise=True)`). Reader / writer:
  tagged implicit-sync NoC transfers per ring entry, tag handed back as the last statement. Compute: one
  `eltwise_chain` per block — `CopyTile` a → D0, `CopyTile` b or `FillBitcast` (number) → D1, `DivBinary`,
  `Floor` / `Trunc` (CT `rounding_mode`), `PackTile` into the output dtype; regime chosen by the define
  `DIVISOR_IS_NUMBER`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  memory_config=[DRAM, L1], sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned],
  rank=[2, 3, 4], broadcast=[none, number], rounding_mode=[None, floor, trunc],
  output_dtype=[None, bfloat16, float32] (= the specification's PHASE0)
- **Accuracy achieved** (`test_div_precision_baseline.py`, 4 shapes × 2 output dtypes + floor/trunc):
  bfloat16 output PCC ≥ 0.9999987, rel RMS ≤ 1.7e-3, **0 ulp** on every element; float32 output (bf16
  operands) PCC = 1.0000000, rel RMS ≤ 1.0e-7, ≤ 2 fp32 ulp; floor / trunc exact (0 mismatches);
  got/true ratio median 1.0 (no scale bug).
- **Golden suite at Phase 0**: 435 / 435 supported cells passing (432 PHASE0 + 3 regression), 2 / 2 rejections;
  4608 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Issues encountered**:
  - Implementer: bf16 outputs met the 0-ulp grade by packing the fp32 quotient directly; no in-DEST RNE
    narrowing was needed.
  - Verifier code-review fixes (host only, `div.py`): an `int` divisor past the float range now saturates to
    ±inf instead of raising `OverflowError`; a numpy scalar divisor is accepted as a number
    (`numbers.Real`); `compute_kernel_config=None` resolved through one helper that calls
    `default_compute_kernel_config()` (R3), and `memory_axes(a)` computed once. Golden re-run identical.
- **Performance**: not measurable on ttsim (no clock). Structural counts: each tensor crosses DRAM once, the
  number 0 times; up to 32 clusters × 4 NEOs engaged; one chain call per `BLOCK_TILES` tiles per NEO.
- **Tests added**: `test_div.py` (acceptance, generated), `test_div_thread_plan.py` (every NEO ran, plans ×
  block sizes with ragged blocks, buffer depths, number path, R8 / number kinds, R13 placement, device-session
  ordering incl. `ttnn.to_layout` after div), `test_div_precision_baseline.py`.
