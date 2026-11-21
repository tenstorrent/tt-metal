# Changelog: scatter

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `resident_line_block` (op_design.md): one Metal 2.0 `ttnn.ProgramSpec` dispatch, one DM kernel with thread plan (6, 0, 6), no compute kernel, no dataflow buffer. A width split (face-columns) across all clusters keeps the whole output line of each block in L1.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved], dim=[-2], index_form=[lines], alignment=[tile_aligned, h_non_aligned], rank=[2] (= PHASE0)
- **Accuracy achieved**: PCC=1.000, max_abs_err=0.0, rms_err=0.0, 0 ULP: bitwise exact, measured on 5 shapes via test_scatter_precision_baseline.py, plus the model shape [2766, 2560] / Hs 2752 by probe
- **Golden suite at Phase 0**: 12 / 12 supported cells and loose cases passing, 4 / 4 rejections, 223 xfail_expected, 0 loud (per `golden_results/verifier_report.json`); queue_gaps 0
- **Issues encountered**: no drift. Verifier code-review fixes:
  - `SRC_STAGE_DEPTH` now reaches the kernel as a CT arg, guarded by a `static_assert` until the ping-pong is built.
  - The host checks the index page geometry the kernel relies on (`index_page_bytes == W·4`).
- **Tests added**: test_scatter.py (acceptance, generated), test_scatter_plan.py (threads ran, one session, forced knobs, R6, rank-3 batches, host clamps), test_scatter_precision_baseline.py

### Implementer notes (Phase 0)

- Host program: `scatter_program_artifacts.py`. Kernel: `kernels/scatter.cpp` (`num_threads = SCATTER_THREADS = 6`).
- Work split: (image, face-column) units over all clusters with `split_work_to_cores(row_wise=True)` (traced call: 32 of 32 clusters). `choose_blocking` sets the knobs `block_width_faces` and `block_src_tile_rows`, one CT arg each.
- The batch axis is already in the kernel: rank 3/4 with `dim` -2 is tested through the program directly (`test_images_along_rows`) but not claimed in SUPPORTED.
- Departures from the design (advisory parts):
  - Each scratchpad region starts on its own 64 B cache line (64 B slack per scratchpad).
  - The slab's `scoped_lock` is one RAII scope across the chunk loop, with one extra `sync_threads` per block before the write-out.

### Not verified on this backend

- ttsim is functional. Whether it models the DM harts' write-back caches is not established, so the cache protocol is correct by construction, not shown by a test.
- No timing. Structural counts are in `verification_report.md` (*Structural counts*). The thread-plan alternatives need `ttsim-perf` or hardware.
