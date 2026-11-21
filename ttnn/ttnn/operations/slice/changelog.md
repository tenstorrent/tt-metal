# Changelog: slice

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `page_relay`: whole interleaved pages relayed through one DM→DM DFB ring per cluster over the full 8×4 grid, page map on the reader, contiguous output pages on the writer, one `ttnn.generic_op` (Metal 2.0 `ProgramSpec`) dispatch; implicit sync for pages ≤ 8 KiB, explicit block-batched sync above (implementer, after a measured ISR miscount on larger entries). Thread plan (R, W) = (1, 1), no compute kernel.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1], sharding=[interleaved], sliced_region=[outer, h], start_alignment=[tile_aligned], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3]
- **Accuracy achieved**: bit-exact — PCC=1.000, max_abs_err=0.0, rms_err=0.0, got/true ratio ≡ 1 (measured on 5 shapes via test_slice_precision_baseline.py, incl. the explicit-sync path)
- **Golden suite at Phase 0**: 69 / 69 supported cells passing (68 PHASE0 + loose `bounds_as_lists`), 412 xfail_expected, 3 / 3 rejections, 0 loud (per `verifier_report.json`)
- **Issues encountered**: implementer — implicit-sync ring entries > 8 KiB (one NoC overlay packet) miscount on the DM0 ISR → explicit sync for large pages. Verifier — no drift; DRY fix of the ring accessor name; added a session-order test after the explicit-sync path; recorded the 8 KiB cap and the implicit-sync txn-id batching rule (batch = half an even ring regardless of `RING_DEPTH_BLOCKS`) in `op_design.md`; ledger audit appended.
- **Tests added**: test_slice.py (generated acceptance), test_slice_plan.py (thread plan, session order, R7), test_slice_precision_baseline.py
