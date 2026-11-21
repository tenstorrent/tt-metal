# Changelog: nlp_concat_heads

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `tile_relay`: reader → one implicit-sync ring (`relay`) → writer, no compute kernel (plan (R, C, W) = (1, 0, 1)),
  flat output-page split over the whole grid (`split_work_to_cores`, `row_wise=True`); one `ttnn.generic_op` of one
  `ttnn.ProgramSpec`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved],
  output_memory=[DRAM], alignment=[tile_aligned], batch=[single] (= the specification's PHASE0)
- **Accuracy achieved**: exact — PCC=1.0, max_abs_err=0, rel_rms_err=0, 0 ULP, got/true ratio 1.0 everywhere
  (4 bf16 shapes through the entry point, 3 fp32 shapes through the program; test_nlp_concat_heads_precision_baseline.py)
- **Golden suite at Phase 0**: 10 / 10 supported cells passing; 206 xfail_expected; 0 supported_fail / xpass_drift /
  xfail_wrong_mode; rejections X1–X3 pass (per `verifier_report.json`)
- **Issues encountered**: verifier fixes, no drift: (1) the program builder now refuses `head_dim % 32 != 0`
  (`NotImplementedError`, regime `subtile_assemble` not built) instead of flooring `D/32`; (2) the L1 ledger's
  live set corrected for multi-hart plans.
- **Tests added**: test_nlp_concat_heads_plan.py (implementer: walk wraps, knobs, engaged cores, one device
  session, rejections; verifier: regime guard), test_nlp_concat_heads_precision_baseline.py

### Implementer notes (Phase 0)
- Knobs, each defined once in `nlp_concat_heads_program_artifacts.py`: `THREAD_PLAN` (1, 0, 1),
  `RING_DEPTH_TILES` 8, `BLOCK_TILES` None (= the core's whole range, one block), `MIN_TILES_PER_CORE` 1.
  Every one is exercised at non-default values in `test_nlp_concat_heads_plan.py` and gives the same bytes.
- Structural counts (simulator: no timing is a result here): each input tile crosses DRAM once, each output
  tile once; 0 cross-core bytes; one NoC transaction per tile; up to `RING_DEPTH_TILES` reads in flight
  per core; clusters engaged = min(32, tiles) (8 for [1,4,32,64], 16 for [1,4,32,128], 32 from 32 tiles).
- `ttnn.add` cannot serve as the "other program" on Quasar (Metal 1, refused at kernel creation); the
  one-session test uses tt-metal's tilize / untilize instead.
