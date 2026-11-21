# Changelog: add

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `stream_flat` of op_design.md: Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` per call, Quasar on ttsim.
  - Work distribution: flat page ranges over the full grid (`split_work_to_cores`, row_wise, in rounds of C
    pages); clusters engaged = min(num_tiles, 32) (tested: 13, 52, 263, 1050 tiles).
  - Thread plan (1, 1, 1) from host constant `THREAD_PLAN` (data-movement bound); plans (2,2,2), (4,4,2), (1,4,1)
    tested correct with a NaN-prefilled output (a missing thread leaves NaN pages).
  - Knobs `BLOCK_TILES`=4, `RING_DEPTH_BLOCKS`=2, `MIN_TILES_PER_CORE`=1 are live and test-turned (block 1/2/4
    with ragged tails, depth 2/3/4, min tiles 1/4/8). The host also clamps `block_tiles` to the config's DEST
    section (mirrors `dest_helpers.hpp`), so a turned knob cannot trip the kernel's static_assert.
  - Reader/writer: implicit sync, one tagged transfer per page, tag handed back; session test with tt-metal's
    `to_layout` before and after passes.
  - Compute: one `compute_kernel_lib::add` call per NEO (PerBlockSize wait/pop/reserve/push, Block mapping).
  - L1: 3 rings, 49,152 B; a, b, out cross their tier once each (`l1_ledger.md`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1],
  sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  broadcast=[none], output_dtype=[None]
- **Accuracy achieved**: PCC=0.999998, max_abs_err=0.0156, rel_rms_err=0.0018, ≤ 1 bf16 ulp (≈1e-5 of elements at
  1 ulp, rest 0), got/true ratio median 1.0000 (measured on 4 shapes via test_add_precision_baseline.py)
- **Golden suite at Phase 0**: 87 / 4694 passed (86 supported_pass + rejection X1), 3046 xfail_expected,
  1560 skipped (INVALID / unbuildable bf8b), 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode, 0 hangs
  (per `verifier_report.json`)
- **Issues encountered**: verifier fixes only — replaced an unused `NEOS` constant and a duplicated `(1, 2, 4)`
  literal with one `COMPUTE_THREAD_COUNTS`; corrected two stale `l1_ledger.md` symbol-table bounds
  (`block_tiles` DEST clamp, `RING_DEPTH_BLOCKS` predicate). No drift fixes.
- **Tests added**: test_add.py (generated acceptance), test_add_plan.py (thread plans, knobs, session, structural
  counts, refusals), test_add_precision_baseline.py
- **Performance**: no time figure — ttsim has no clock; perf is structural on this backend.
