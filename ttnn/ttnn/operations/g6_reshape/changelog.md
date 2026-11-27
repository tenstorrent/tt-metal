# Changelog: g6_reshape

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier, two verifier passes). Output-stationary face-row gather: one `generic_op`, one data-movement kernel source run as two movers per core (RISCV_1/NoC0, RISCV_0/NoC1), gather → await → permute → store per block. Regimes: `tile_face_row` and `tile_element_run`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE_LAYOUT], DRAM interleaved, rank 1..4 on both sides, any last two dims. The second verifier pass added float32: the tile kernel already moved 4-byte elements bit-exactly, and TILE is float32's production layout.
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, max ULP=0, got/true ratio exactly 1.0. Measured on 5 shapes with `test_g6_reshape_precision_baseline.py`: 3 bf16 `tile_face_row`, 1 bf16 `tile_element_run`, 1 fp32 `tile_element_run`.
- **Golden suite at Phase 0**: with `EVAL_CAPTURE_L1=0`, 28 / 28 supported cells pass, out of 59 total (25 `xfail_expected`, all int32/uint32 ROW_MAJOR; 6 regression tests without axes, all passing). In the graded default run, 27 / 28 pass; the trace cell and 4 validation regression tests fail for harness reasons H1/H2 (`verification_report.md`). Both runs show 0 xpass_drift and 0 xfail_wrong_mode.
- **Issues encountered**: none in the kernel. Verifier fixes, first pass:
  - one sizing helper feeds the BB-05 cap, the CB sizes and the plan report;
  - one `MOVERS` placement table;
  - new `MAX_BLOCK_ATOMS` knob;
  - knobs added to the descriptor-cache key (BB-07);
  - tile-table `--dev` ASSERT;
  - `max_blocks_per_mover` in `describe_plan`;
  - dead constant removed.

  Second pass:
  - float32 added to SUPPORTED on probe + golden evidence (7 cells xfail → pass);
  - plan-CB tile table sized by `block_atoms` instead of `block_atoms·atom_rows`. The ledger had claimed the wrong bound; bf16 L1 peaks drop, e.g. 12672 → 12416 B on `(1,1,16,4608)→(16,1,4608)`;
  - ledger and design CB table corrected;
  - acceptance and precision tests made dtype-generic, with fp32 FLUX, fp32 special-value and fp32 multi-block knob coverage.
- **Tests added**:
  - `test_g6_reshape.py`: acceptance, including `test_block_knobs` (multi-block, depth-2, one-mover and one-core paths, bf16 and fp32) and `test_float32_*`;
  - `test_g6_reshape_debug.py`;
  - `test_g6_reshape_precision_baseline.py`.
- **Perf reference (profiler build, device kernel ns)**:
  - bf16 `(16,1,5120)→(1,1,16,5120)` 8783, `(1,1,16,4608)→(16,1,4608)` 6974.
  - fp32 FLUX cells 1.8–2.2 µs on 110 cores.
  - Small bf16 cells ≈ 2.0 µs on 8–48 cores.

  The mover is RISC-V-cycle-bound, about 250 ns per output face-row (zone profile in `verification_report.md`).

## Refinement 1 — Integer ROW_MAJOR regime (int32, uint32; ROW_MAJOR)
- Date: 2026-11-27
- What was done:
  - SUPPORTED: dtype += int32, uint32; layout += ROW_MAJOR_LAYOUT (one change, per the spec rule). No EXCLUSIONS.
  - New regime `rm_row_gather`, chosen on the host for ROW_MAJOR inputs; still exactly one `generic_op` per call. Kernel `kernels/g6_reshape_rm_mover.cpp` uses the same skeleton (gather → await → permute → store, two movers per core on RISCV_1/NoC0 and RISCV_0/NoC1).
    - Atom = one `A`-byte (DRAM alignment) chunk of an output page slot.
    - Atoms are folded bank-affine on the device, ordered by (bank, page/NB, chunk) using `NUM_DRAM_BANKS`, so a mover's block is consecutive bytes of one bank. It stores as one write per (block, bank).
    - Reads are A-aligned covering runs inside one input page, with a read-through gap of up to `RM_READ_GAP_BYTES` (new knob, default `A`, part of the cache key). For `[1,W]→[W,1]` each mover's strided elements come in as one read.
    - Permute is a strided copy: one plan entry per input-row piece, no per-element 64-bit math. It is generic over element sizes 2 and 4, so bf16/fp32 ROW_MAJOR work through the same code.
  - Reused: the mover skeleton, `MOVERS` placement, block knobs, descriptor cache and `describe_plan`. The core/mover split is now `_split_atoms`; the BB-05 block fit is now `_fit_block_atoms`, called by both regimes; the core-range and stage/image byte logic is in `_CoreRangesMixin`. The permute block operation was moved to `kernels/g6_reshape_permute.hpp` and is shared by both kernels; the tile kernel's behavior is unchanged.
  - Added: `_RmPlan`, `_rm_slot_bytes` (one source for the BB-05 cap, the CB sizes and `l1_bytes_per_core`), `_rm_compile_args`, the RM kernel, and RM rows in `l1_ledger.md`.
  - Why a second kernel source: the RM gather walk and store geometry share nothing with the tile path, and the compile-time-arg layouts don't overlap. The shared operation (permute) is shared code.
- **Harness fix (`tests/plugins/up_front_collect.py`)**: this hang is what failed the previous attempts' gates. It reproduces on the untouched Phase 0 tree and with a stock `ttnn.relu`.
  - The precompile warmup runs test bodies under NO_DISPATCH, with `torch.randn` replaced by zeros and shape-only `from_torch`.
  - The bf16 trace loose case's eager exact check can then pass on zeroed DRAM. `run_g6_reshape` goes on to capture an empty trace and `execute_trace` it, which hangs the device.
  - The plugin's `_drives_capture` only scans the test function's own source, so it misses the helper's trace path. After the warmup hang, the graded run errored on all 59 cells.
  - Fix: inside the collect window, the plugin's capture markers (`begin_trace_capture`, …) now raise. The body aborts after its eager ops are stashed, the exception is swallowed like any collect-pass failure, and the trace still compiles cold in the real pass. This is the skip behavior the plugin's docstring already describes.
- Accuracy achieved: bit-exact (exact gate) on every case. That includes full-range int32 (`int32_extremes`), uint32 with the high bit set, bf16/fp32 ROW_MAJOR, and int32/uint32 TILE.
- Golden test progress, harness runner `eval/eval_test_runner.sh` with precompile, graded default:
  - 54/59 passed, 0 hangs.
  - All 25 previously `xfail_expected` cells pass: 15 int32 RM, 5 uint32 RM, and 5 loose cases.
  - The 5 failures are the known harness caveats from Phase 0: H1 (the bf16 trace cell under `EVAL_CAPTURE_L1=1`: "program not in program cache during trace capture") and H2 (4 `test_validation_errors`, L1 peak 0).
- Perf baseline for Refinement 2 (profiler build, device kernel ns, 110 cores): `[1,4096]→[4096,1]` int32 **4610 ns**; `[1,3844]→[3844,1]` int32 4280 ns. Incumbent: 362.12 µs.
- Issues encountered: the precompile trace hang above. No kernel issues; first device run was bit-exact.
- Tests added: `test_g6_reshape_row_major.py`, 49 cases, plain and `--dev`:
  - every int32/uint32 RM production shape and the `[16]→[4,4]` sequence;
  - bf16/fp32/uint32 RM general geometries, including odd widths, wide rows and column→row;
  - int32/uint32 TILE;
  - the flagged-plan pin (`regime == rm_row_gather`, `num_cores > 1`, within the L1 budget) with extremes and `-1`;
  - an RM block-knob matrix: multi-block, depth 2, one mover, one core, no read gap.
