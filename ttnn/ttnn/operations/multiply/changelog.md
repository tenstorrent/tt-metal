# multiply changelog

## Phase 0 (ttnn-implementer)

Built as designed (op_design.md), no deviation from the binding parts:

- One `ttnn.ProgramSpec` per call via `ttnn.generic_op`; regime chosen on host by `select_regime`
  (`flat_stream` for broadcast `none`, `row_bcast_strips` for `over_h`), passed as CT arg `regime`.
- Knobs, each defined once in `multiply_program_artifacts.py`: `THREAD_PLAN = (1, 1, 1)`, `STREAM_DEPTH = 4`,
  `B_WINDOW_DEPTH = 2`, `L1_RESERVE_BYTES = 0`; per call `row_groups` (G) and `block_cols`
  (`min(Wt, max_units_per_core, block_cols_cap)`). Ring capacities derive from them; the block decomposition
  lives in one header (`kernels/multiply_block_walker.hpp`) used by all three kernels.
- Reader/writer implicit sync (tagged transfers, tag handed back at kernel end); the `row_bcast_strips` b
  window is explicit sync with user trid 0, nominal `block_cols` windows, listed in the reader's
  `disable_dfb_implicit_sync_for`.
- Compute: `compute_kernel_lib::mul` (Scalar/PerTile streaming; `BroadcastDim::Row` + `InputTileMapping::Row`
  on the held window, caller wait/pop of `block_cols`). No raw LLK.
- Shape-dependent values (G, Ht, Wt, b's image strides) are common runtime args, so cells of one regime share
  compiled kernels; `block_cols` and the plan are CT args as the design says.

Structural counts (ttsim: no timings are performance results): every tensor crosses the DRAM/L1 boundary
once, except b in `row_bcast_strips`, which crosses G times (G = 2 only for `[128,512]x[1,512]` among INPUTS);
one helper call per block; blocks per core = 1 on the YOLO shapes (`[1,4,8400]`: 9 or 8 columns per
cluster, 32 clusters).

Tests: acceptance 13/13; `test_multiply_regimes.py` 17/17 (regime pinning incl. G = 2, multi-block and
short-block runs, leading-dim broadcast of b, distinct-column broadcast, flat_stream at plans (2,2,2),
(4,4,2), (1,4,1), one device session with a tilize before and an untilize after, R9 placement, X1/R2
refusals); golden PHASE0 slice 99/99 (incl. numeric regression and X1), 40 sampled out-of-phase cells xfail.

## Phase 0 — Core Implementation (verified)
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1],
  sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  broadcast=[none, over_h], output_dtype=[None, bfloat16]
- **Accuracy achieved**: bit-exact (0 ulp, 100% of elements) on 6 shapes; vs the float64 product
  PCC ≥ 0.9999986, max_abs_err ≤ 0.0312, rel_rms_err ≤ 1.66e-3 (the one bf16 rounding), got/true ratio
  median 1.0000 (measured via test_multiply_precision_baseline.py)
- **Golden suite at Phase 0**: 98 / 98 supported cells passing (92 PHASE0 + 6 numeric regression), X1 passing,
  1540 refused as expected; 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)
- **Issues encountered** (verifier fixes, no drift):
  - the compute config now reaches the kernel whole: `math_approx_mode` → `sfpu_precision_mode`,
    `dst_full_sync_en` → `double_buffer_dest` (it was dropped);
  - `block_cols_cap` sizes each ring with its own page size (it used the output's for all three, wrong once
    dtypes differ); `l1_ledger.md` updated.
- **Tests added**: test_multiply_precision_baseline.py; `test_compute_config_passed_through` in
  test_multiply_regimes.py
- **Queue**: op_requirements.md — R1 ROW_MAJOR, R2 broadcast completeness, R3 perf (YOLO first-output latency),
  R4 numeric (float32, fp32_dest_acc_en=False, float32 output), R5 perf (leading-dim b reuse), R6 perf
  (per-block fixed costs); perf gates are structural counts on ttsim.
