# Changelog: zeros_like

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4] (= spec PHASE0)
- **Accuracy achieved**: exact — PCC=1.0, max_abs_err=0.0, rms_err=0.0, 0 ULP, no −0.0 (measured on 4 shapes ×
  {normal, NaN/±inf} inputs via test_zeros_like_precision_baseline.py)
- **Golden suite at Phase 0**: 21 / 21 SUPPORTED cells passing, 147 xfail_expected, 0 loud (per `verifier_report.json`)
- **Issues encountered**: verifier fixes: `validate()` names every axis outside SUPPORTED, so a sharded input is
  refused by `sharding` (R5); the R5 sharded-input test was added; the `pages_per_barrier` row of `l1_ledger.md`
  was corrected to match the code.
- **Tests added**: test_zeros_like.py (harness), test_zeros_like_plan.py (implementer; verifier added
  `test_refuses_sharded_input_by_name`), test_zeros_like_precision_baseline.py (verifier)

### Implementer notes (Phase 0)

Built as op_design.md says: regime `fill_from_local_zero_page`, one `ttnn.generic_op` dispatch of a Metal 2.0
`ProgramSpec`, thread plan (R, C, W) = (0, 0, 1) from the host constant `THREAD_PLAN`.

- `zeros_like.py`: `INPUT_TAGGERS = AXES` (spec), `SUPPORTED` = spec `PHASE0`, `EXCLUSIONS = []`, `validate()`
  with `eval.memory.memory_axes` for the placement axes (R5); a zero-volume tensor is refused (no program to
  dispatch). Output from `ttnn.allocate_tensor_on_device(tensor.spec, device)` (R2, R3, R4).
- `zeros_like_program_artifacts.py`: writer-only work unit over `split_work_to_cores(..., row_wise=True)`
  contiguous page ranges; `zero_page` `ScratchpadSpec` of `W * zero_src_bytes`; knobs `PAGES_PER_BARRIER`
  (256), `ZERO_SRC_MAX_BYTES` (64 KiB), `THREAD_PLAN`; `pages_per_barrier` clamped by the trid counter cap.
- `kernels/zeros_like_writer.cpp`: `zero_source` (iDMA zero + `write_zeros_l1_barrier`, once per kernel),
  `store_block` (`TXN_ID` writes with trid 0, one trid barrier per block, ragged last block), `release_tag`.

Deviation from the task's file list: there is no `zeros_like_reader.cpp` and no `zeros_like_compute.cpp`. The
design's pipeline topology (binding) has no reader (R1: the input is never read) and no compute kernel.

Findings:
- The iDMA zero mode works on ttsim (no fallback to a CPU fill was needed). Verifier's probe_006 confirms it is
  load-bearing: without it, stale L1 from a previous program reaches every output element.
- The "tt-metal program after ours" test uses the `toy_thread_plan` example as the program
  before; zeros_like right after softmax is tested separately and passes (its writes are all tagged).
- Structural counts (Phase 0, 32 clusters): for every INPUTS shape each core runs exactly one block (≤ 32
  pages ≤ PAGES_PER_BARRIER), so one write barrier and one iDMA zero per core; 1 NoC write per page; 0 bytes
  read; output crosses DRAM once. W = 2 is verified correct (test_every_page_written[W2]) but not adopted:
  this backend has no timing, and the target page's rule for a data-movement-bound op is one thread.
