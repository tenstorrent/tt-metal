# Changelog: plus_one

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier) for Quasar on ttsim. One `ttnn.generic_op` dispatch of a Metal 2.0 `ProgramSpec`: a single data-movement kernel (`WORKER_THREADS = 4` threads per engaged cluster) that reads its row segments into a line-aligned scratchpad region, increments them as `uint32_t` under one `scoped_lock`, and writes them back in place (Regime A `segment_split_in_place`). No compute kernel, no DataflowBuffer, no output allocation (R3).
- **SUPPORTED at Phase 0**: dtype=[int32, uint32], layout=[ROW_MAJOR], memory_config=[DRAM], sharding=[interleaved], skip_negative_entries=[False, True], rank=[1] (= PHASE0)
- **Accuracy achieved**: exact — PCC=1.000000, max_abs_err=0, rms_err=0, max ULP=0 (measured on 4 shapes × 2 dtypes × 2 flag values via test_plus_one_precision_baseline.py)
- **Golden suite at Phase 0**: 28 / 28 supported cells passing; 148 xfail_expected; 0 loud (per `golden_results/verifier_report.json`)
- **Issues encountered**: no drift. Verifier code-review fixes: (1) host common-runtime-arg schema derived from the values dict instead of a duplicate name list; (2) kernel unit → segment mapping defined once (`segment_of`) instead of three copies across load / increment / store. L1 ledger: verifier note on the `SCRATCH_DEPTH > 1` dead-capacity case.
- **Tests added**: test_plus_one.py (harness-generated acceptance), test_plus_one_plan.py (thread plan, session, pinned splits, block path), test_plus_one_precision_baseline.py
