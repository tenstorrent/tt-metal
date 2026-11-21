# Changelog: clone

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), for quasar on ttsim. Built as designed (`op_design.md`, regime `interleaved_page_stream`).
  - Host: `clone_program_artifacts.py` builds a Metal 2.0 `ProgramSpec`, dispatched once with `ttnn.generic_op`. Knobs are single-sourced: `THREAD_PLAN = (1, 1)`, `RING_ENTRIES_PER_THREAD = 4`, the grid is queried, `entry_bytes` is derived, and pages are dealt to clusters in whole rounds of R (`split_pages`, `row_wise=True`).
  - Kernels: `clone_reader.cpp` and `clone_writer.cpp`, with implicit sync at both ends of the DM→DM ring `staged_pages` and the trid handed back last. There is no compute kernel (`compute_passthrough` is rejected in the design).
- **SUPPORTED at Phase 0**:
  - dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1], sharding=[interleaved]
  - dtype_argument=[None, "same"], output_memory=["L1"]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4]
  - (= PHASE0)
- **Accuracy achieved**: bit-exact, measured on 4 shapes via `test_clone_precision_baseline.py`: PCC=1.000, max_abs_err=0.000, rms_err=0.000, ULP 0, got/true ratio exactly 1.
- **Golden suite at Phase 0**: 200 / 200 supported cells pass, plus the X1/X2 rejections. 1000 xfail_expected (float32; output_memory None/DRAM). 0 supported_fail / xpass_drift / xfail_wrong_mode, queue_gaps 0 (per `verifier_report.json`).
- **Structural counts** (the perf measure on this backend; time gains unconfirmed until a timed backend):
  - 1 tagged read + 1 tagged write per page;
  - 0 barriers / reserve / push / wait / pop in either kernel;
  - each tensor crosses its tier once;
  - clusters engaged = min(32, tensor_pages): 32 for every traced YOLOv8 call;
  - ring 8 KiB per cluster for tiles.
- **Issues encountered** (verifier):
  - The page-size mechanism-cap refusal now raises the typed `SupportRefusal`, not a bare `NotImplementedError`; a test was added for it.
  - L1 ledger fixes: the depth bound now matches the code, and the `owner_local` deferral reason is corrected (`ttnn.reports.get_buffer_pages` exposes page→core). That split is queued as perf Refinement 2.
  - No drift fixes were needed.
- **Tests added**:
  - `test_clone.py` (harness-generated acceptance, 25);
  - `test_clone_plan.py` (24): every thread ran at R = 1, 2, 3; device-session order before and after a tt-metal program; R3 placement; R5 fresh buffer; X1/X2/R7/R8 refusals; oversized-page refusal;
  - `test_clone_precision_baseline.py` (4).
