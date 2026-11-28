# Changelog: g6_copy

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  - Whole-page DRAM → L1 → DRAM copy over the full grid: a reader on NCRISC/NoC0 and a writer on BRISC/NoC1.
  - One staging CB. Transfers are coalesced bank runs with bank stagger on.
  - `one_block` plan by default, with a `streamed` fallback under the L1 clamp.
- **SUPPORTED at Phase 0**:
  - dtype=[float32]
  - layout=[TILE]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned]
  - EXCLUSIONS (contract): {float32, ROW_MAJOR}, {int32, TILE}, {uint32, TILE}
- **Accuracy achieved**: bit-identical on 4 fp32 shapes, (1, 1) to (16, 12, 128, 128), measured with
  `test_g6_copy_precision_baseline.py`: PCC=1.0, max_abs_err=0.0, rms_err=0.0, 0 ULP, got/true ratio 1.0.
- **Perf at Phase 0**: (16, 12, 128, 128) fp32 runs at 67.6 µs median device kernel time (goal < 68.56 µs,
  roofline 54.61 µs).
- **Golden suite at Phase 0**: 13 / 42 passed (per `verifier_report.json`):
  - 10 / 39 registry cells supported_pass.
  - 29 xfail_expected.
  - 0 loud.
  - 3 / 3 numerics regression tests pass.
- **Issues encountered** (verifier):
  - No op or kernel fix needed.
  - Deleted the stale untracked `test_g6_copy_perf.py`, `ablation_kernels/` and `test_g6_copy_knob_sweep.py`.
    They were left by an earlier implementer session; the perf test patched symbols the current descriptor
    does not define, failing 24 tests in a directory run.
  - The local conftest now keeps a module's own `use_module_device(...)` marker.
  - `op_design.md` updated to match the code's `BANK_STAGGER = True` default.
- **Tests added**: `test_g6_copy_precision_baseline.py`, `test_g6_copy_trace.py` (BB-16 trace replay). The
  acceptance test `test_g6_copy.py` and perf sweep `test_g6_copy_lamps.py` come from the implementer.

## Refinement 1 — Full dtype set and ROW_MAJOR layout
- Date: 2026-11-28
- What was done: SUPPORTED widened to dtype {float32, bfloat16, bfloat8_b, int32, uint32} and layout
  {TILE, ROW_MAJOR}. No kernel or descriptor change: the page walk is already the pure-RM data-movement
  pattern (a page is a row, `page_bytes = buffer_aligned_page_size()`), and the CB format is the tensor's own,
  never interpreted. The three contract EXCLUSIONS ({float32, RM}, {int32, TILE}, {uint32, TILE}) kept;
  bfloat8_b × RM stays INVALID (feature_spec). New paths stay on the full-grid page split (plan test asserts
  `num_cores = min(pages, grid)`). Reused: SUPPORTED/validate(), the reader/writer pair, descriptor. Added:
  only SUPPORTED values and one test file.
- Accuracy achieved: bit-exact (float32 / bfloat16 / int32 / uint32) and within the bfp8 one-step gate
  (bfloat8_b) on every production row, adversarial bits per dtype and bf16 poisoned padding.
- Golden test progress: the 29 former xfail cells (24 production rows + 5 loose cases) pass (golden slice
  `-k "not float32"`: 29/29); the 10 float32 cells were untouched.
- Issues encountered: None.
- Tests added: `tests/ttnn/unit_tests/operations/g6_copy/test_g6_copy_dtypes_layouts.py` (values per
  non-float32 production row incl. shortest RM rows uint32/int32 `(1,)`, adversarial bits per dtype, bf16
  poisoned padding (1, 1, 1, 256), `describe_plan` determinism / full-grid / L1 per row).

## Refinement 2 — Speed up the perf-flagged profile: overlap reads and writes
- Date: 2026-11-28
- What was done: knob-turn on the existing `streamed` plan. New single-source knob `BLOCK_BYTES = 16384` in
  `g6_copy_program_descriptor.py`: `block_target = BLOCK_BYTES // page_bytes` (capped by the forced-test
  `MAX_BLOCK_PAGES`). `_plan()` keeps `one_block` when the per-core assignment fits one block (so 1–4 page/core
  shapes get no extra hand-offs) and otherwise streams `block_target`-page blocks through a
  `STREAM_DEPTH_BLOCKS = 2` ring, so the writer drains block k while the reader fills block k+1. `BLOCK_BYTES = None`
  restores the Phase 0 coarsest-block plan byte-for-byte. No kernel change (the reader/writer already block both
  halves symmetrically). Knob in the descriptor cache key and reported by `describe_plan()`.
  Reused: `_plan()`, the streamed path and both kernels. Added: one knob, one perf sweep file.
- Measured (device kernel ns, Blackhole 11x10, fresh-cache, median of 5): (16, 12, 128, 128) fp32 68.0 → 61.6 µs
  (samples 60.5–61.7); (8, 12, 128, 128) fp32 32.9 → 30.8 µs; bf16 (1, 1, 32, 62080) 21.3 → 19.2 µs (single run);
  (2|1, 12, 128, 128), bf16 (1, 1, 8448, 128), bf8b (5, 1, 32, 256), bf16 RM (16, 3, 2560) within noise.
  Depth 3 / 8 KiB / 24 KiB / 32 KiB measured, none better beyond noise across both flagged shapes.
  Remaining headroom (61.6 vs 54.61 µs roofline): per-block `noc_async_read_barrier` hand-off and 4 KB runs —
  Refinement 3 levers.
- Accuracy achieved: bit-exact (fp32/bf16/int/uint), bfp8 one-step gate, on every guard shape and variant.
- Golden test progress: slice `-k "float32 or bfloat16 or loose"` 30/30; unit directory green.
- Issues encountered: `test_g6_copy_plan` pinned `one_block`; relaxed to the BB-08 contract (default plan per
  `BLOCK_BYTES`). Stagger / lamp tests now pin `BLOCK_BYTES = None` since they measure the one-block plan; the
  forced streamed test sets it to None so `MAX_BLOCK_PAGES` alone sets the extent.
- Tests added: `tests/ttnn/unit_tests/operations/g6_copy/test_g6_copy_overlap.py` (BLOCK_BYTES × depth sweep over
  the guard set, 5-repeat samples on the flagged shapes).
