# Changelog: all_gather

## Phase 0 — Core Implementation
- **Date**: 2026-07-03
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL op:
  `generic_op` + `MeshProgramDescriptor` running a bidirectional store-and-forward
  ring on newly-authored dataflow kernels (`all_gather_reader.cpp`,
  `all_gather_writer.cpp`). Two worker cores per device (`core_fwd`, `core_bwd`),
  three single-owner GlobalSemaphores (barrier + fwd/bwd counting), fabric egress via
  the `ccl_helpers_dataflow.hpp` safety-by-construction helper.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16, float32]
  - layout = [TILE]
  - topology = [Linear]
  - gather_dim = [-4]  (negative convention; -4 ≡ dim 0 for rank-4 shards)
  - alignment (INPUT_TAGGERS) = [tile_aligned, non_tile_aligned]
  - EXCLUSIONS = []
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0.0, mean_abs_err = 0.0,
  rel_rms_err = 0.0 — bit-exact identity gather across 4 shapes × {bf16, f32}
  (measured via `test_all_gather_precision_baseline.py`). All 8 devices agree
  bit-for-bit (replicated output).
- **Golden suite at Phase 0**: **16 / 384 cells passing** (`supported_pass = 16`,
  `xfail_expected = 304`, `invalid_skipped = 64`; loud categories `supported_fail`,
  `xpass_drift`, `xfail_wrong_mode` all = 0) — per `verifier_report.json`.
  Run via `scripts/run_multidevice_sim_pytest.py --op all_gather` in 5 dtype/layout
  `-k` chunks (a full single-process golden run exceeds the wall-clock backstop
  because the CCL golden `mesh_device` fixture re-inits the 8-device fabric per cell).
- **Issues encountered / fixed during verification**:
  - Simplified a dead-code branch in `validate()`: `page % 16 != 0 and page != 16`
    → `page % 16 != 0` (the `and page != 16` conjunct was unreachable). Behaviour
    unchanged.
  - Unblocked shared test infra: `tests/ttnn/utils_for_testing.py` referenced
    `ttnn.fp8_e4m3` (from FP8-enablement commit `079872566e`) which the built binary
    predates, breaking collection of every test importing `assert_with_pcc`. Guarded
    the entry with `hasattr(ttnn, "fp8_e4m3")`. Not an all_gather defect.
  - No SUPPORTED drift (`xpass_drift = 0`) — no auto-promotions needed.
- **Tests added**:
  - `test_all_gather.py` (acceptance; pre-existing, 9/9 PASS)
  - `test_all_gather_precision_baseline.py` (pre-existing, 8/8 PASS, bit-exact)
  - `test_all_gather_extended.py` (**new**; preallocated-output path +
    validate() rejection behaviour; 2/2 PASS)
- **Refinement queue set up** (`op_requirements.md`): 3 refinements covering the
  TARGET − SUPPORTED gap —
  1. Format axes: bfloat8_b + ROW_MAJOR (`/memory-layouts`, `/numeric-formats-metal`)
  2. Non-contiguous concat addressing: gather_dim −3/−2/−1 (verifier-authored)
  3. Ring topology (verifier-authored; **verification infra-blocked** — no ring
     topology in the multidevice sim matrix yet)
