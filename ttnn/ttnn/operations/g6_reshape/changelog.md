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

## Refinement 2 — Speed up the perf-flagged int32 row-to-column profile (knob co-tune)
- Date: 2026-11-27
- What was done:
  - **Diagnosis** (device zones, flagged `[1,4096]→[4096,1]` int32, 110 cores × 2 movers): the read barrier (`g6.await`) dominated, at 1.2–1.3 µs median and 2.2–2.5 µs max. The input is ONE 16 KB page in ONE DRAM bank. The bank-affine fold gives each mover output pages NB apart, i.e. input elements 32 B apart, so 220 movers read ≈ 133 KB (8× amplification) from that hot bank.
  - **Lever 1, `RM_FOLD` (new knob, default `"linear"`; read_coalescing + bank grouping)**: atoms fold in stream order, so a block is one contiguous input range and each input byte is read once (≈ 28 KB total). The permute lays the image out bank-grouped (pages `p0+g, p0+g+NB, …` per group), so the store stays ≤ NB writes per block (the design's "~NB writes per mover"). The bank-affine path is kept as `RM_FOLD="bank"` (byte-identical to Refinement 1) and covered in the knob matrix.
  - **Lever 2, pre-issued reads (reader/writer pair with lever 1)**: under the linear fold a block's covering reads (one A-aligned run per input row) are known before the walk. They are issued first, and the walk overlaps their DRAM latency; the walk only re-derives the staging layout and ASSERTs that it matches. Await median 1.2 µs → 0.2–0.3 µs.
  - **Knob co-tune** (fresh descriptor per setting; a warm-up call absorbs a ~200–300 ns first-call-in-module penalty):
    - `MAX_CORES` 8/16/32/64/110: full grid best (64 cores: 3.6 µs).
    - `MOVERS_PER_CORE` 1: 4.4 µs (worse).
    - `MAX_BLOCK_ATOMS=10` at depth 1 or 2: ~4.3 µs (worse; per-block fixed cost).
    - `MOVER_NOCS` (noc_placement, new knob in the cache key; `MOVERS` no longer hardcodes the NoC): swapped placement is ~6% slower on the flagged shape and ~10% slower on tile `(16,1,5120)`, so the native `(NOC_0, NOC_1)` stays the default.
    - `bank_stagger`: no extra mechanism needed. Linear-fold movers start at different banks and write groups in bank order from their own start bank.
  - Reused: the RM mover skeleton, permute block op, cursor/advance, CB sizing (`_rm_slot_bytes`, unchanged bounds) and the knob/cache-key plumbing. Added: the `kLinearFold` CT arg (index 16), group offset/size computation, the pre-issue loop, the group store, and the `RM_FOLD` / `MOVER_NOCS` knobs.
- Perf (device kernel ns, profiler build, 110 cores, warm call):

  | Shape | Bank fold (R1 path) | Refinement 2 | Change |
  |---|---|---|---|
  | `[1,4096]→[4096,1]` int32 (flagged) | ≈ 4250–4400 | **≈ 2900** (2892 / 2898 / 2907 / 2932) | −33%; −37% vs R1's recorded 4610 |
  | `[1,3844]→[3844,1]` int32 | ≈ 4150–4350 | ≈ 2875–2920 | −32% |

  Tile guards are unchanged (no tile-kernel change): `(16,1,5120)→(1,1,16,5120)` 8.5–8.7 µs, `(1,1,16,4608)→(16,1,4608)` 6.83–6.87 µs. Incumbent 362.12 µs.
- Remaining profile (flagged, per mover; median ns):
  - launch to kernel start ≈ 500;
  - issue 100;
  - walk (`g6.gather`) ≈ 500;
  - await 200–300 (max ≈ 800, the hot bank);
  - permute ≈ 620 (19 one-word copies, ~45 cycles each);
  - store ≈ 320 (8 writes);
  - drain ≈ 185.

  The critical path is now the mover's per-unit RISC-V work (walk + permute ≈ 1.1 µs of ~2.3 µs kernel time). That is exactly Refinement 3's scope: plan-free permute for the linear fold, image-free stores, and roles split.
- Accuracy achieved: bit-exact (exact gate) on every case, including `int32_extremes`, the `-1` variant, uint32 high bit, bf16/fp32 RM, and int32/uint32 TILE.
- Golden test progress: targeted slice (`EVAL_CAPTURE_L1=0`, `-k "ROW_MAJOR or 16x1x5120"`) 26/26 pass: all 20 int32/uint32 RM cells, the 5 RM loose cases and a tile representative. The full suite is left to the harness gate.
- Issues encountered:
  - The first sweep's NoC "swap" patched `MOVERS`, which was not in the descriptor-cache key, so it re-ran the default program. The fake "win" was the first-call penalty. Fixed by making the NoC a cache-keyed knob (`MOVER_NOCS`) and adding a warm-up entry to the sweep.
  - The Tracy capture tool sometimes exits with code 3 (no new CSV); re-running is enough.
- Tests added:
  - `test_g6_reshape_perf_sweep.py`: guard-set correctness pin by default; `G6_PERF_SWEEP=1` / `G6_PERF_SWEEP2=1` run the knob sweeps under `--profile`.
  - `test_g6_reshape_row_major.py` knob matrix: `bank_fold`, `bank_fold_block5` and `linear_block7` settings, plus an fp32 `(2,33,40)→(66,40)` case (3 atoms per page; blocks start/end mid-page and span > NB pages). 66/66 pass under `--dev` (kernel ASSERTs on).

## Refinement 3 — Speed up the perf-flagged profile, phase 2 (per-unit mover cost)
- Date: 2026-11-27
- What was done (rm_row_gather, linear fold; tile kernel behaviour unchanged):
  - **Floor check** (Done-when clause): an empty kernel on the same 110 cores measures ≈ 290 ns; a reads-only ablation (R2's per-mover covering reads + barrier, then return) ≈ 1600 ns. R2's ≈ 2900 ns was far from it, so code changed.
  - **Lever 1 — plan-free permute ("store realization"/per-unit cost)**: under the linear fold the staged range is the input stream at a uniform row pitch (row r0's run starts at `lo0`, later rows at 0, every run but the last ends at the pitch), so stream element (r, c) sits at `stage − lo0 + (r − r0)·pitch + c·e`. The gather walk, the plan entries written to L1 and read back, and the plan CB are deleted; the copies run directly off the fold cursor after the barrier, bank group by bank group, each group fenced and written as soon as it is copied. A compile-time fast path for whole-page atoms inside one input row (`[1,W]→[W,1]` family) does one compile-time-sized copy per atom with an incremental in-row index (no per-atom division). Group write addresses are computed between issue and barrier (in the read-latency shadow). Effect: flagged ≈ −5% (the walk had already been overlapped with the DRAM latency by R2), `[1,3844]` −7%. An incremental-cursor variant for the general path measured slower than recomputing divmod per atom and was dropped; a single-fence-before-all-writes variant was within noise and dropped in favour of per-group write/copy overlap.
  - **Lever 2 — shared per-core read (`RM_SHARED_READ`, "mover roles" lamp)**: per-core await profiles showed a strong spatial gradient (90 ns near → 1.4 µs far) and, with NoC1's reads removed, NoC0's await was unchanged: the tail is a per-NoC request queue at the single DRAM bank holding the 16 KB input page. Now one mover per core issues the covering read of the whole core range into a pooled staging CB (mover 0's stage CB, both slots; mover 1's not allocated), and hands it to its partner through a local semaphore. Readers alternate NoCs by core checkerboard → 55 requests per NoC instead of 110. Applies when every mover has a single block (else falls back to per-mover reads). −25%.
  - **Lever 3 — bank split (`RM_BANK_SPLIT`)**: with the core's range staged once, mover m takes bank groups m, m+2, … of the whole core range instead of a stream half, so each issues ≤ NB/2 writes (4 instead of 8 on Blackhole) with the same atom count (exact for even NB; kernel static_assert + ASSERT). Whole-page atoms only. −8%.
  - Shared code: `copy_run`, `begin_image`, `fence_image` factored out of `permute_block` in `g6_reshape_permute.hpp` (tile kernel calls the same `permute_block`, unchanged semantics).
  - Reused: the RM mover skeleton, linear fold / bank-group image layout, `_split_atoms`, `_fit_block_atoms`, `_rm_slot_bytes`, the knob/cache-key plumbing, `describe_plan`. Added: `stream_range`/`issue_range` lambdas, the plan-free copy loop, CT args 17–21 (shared read, semaphore id, bank split, mover index, movers), RT args 2–4 under the shared read (core range, role), `_RmPlan.mover_runtime_args`, knobs `RM_SHARED_READ`, `RM_BANK_SPLIT` (both in the cache key, both reported by `describe_plan`), a semaphore descriptor.
- Perf (device kernel ns, profiler build, 110 cores, warm call, same session):

  | Shape | R2 | + lever 1 | + lever 2 | + lever 3 (shipped) |
  |---|---|---|---|---|
  | `[1,4096]→[4096,1]` int32 (flagged) | 2915 | ≈ 2750–2880 | 2036 | **1878** |
  | `[1,3844]→[3844,1]` int32 | 2897 | 2691 | 2010 | **1844** |
  | bf16 `(16,1,5120)→(1,1,16,5120)` | 8561 | — | — | 8516 (unchanged path) |
  | bf16 `(1,1,16,4608)→(16,1,4608)` | 6821 | — | — | 6812 (unchanged path) |

  Incumbent 362.12 µs.
- Remaining profile (flagged, per mover, median / max ns): issue ≈ 115 (readers only), await 450 / 900 (NoC0 readers' hot-bank tail), permute+store ≈ 490 (19 one-word copies, 4 fences, 4 writes), drain ≈ 190. Remaining headroom is the hot-bank tail: next lever would be fewer readers per NoC (core-pair read + L1→L1 forward) or spreading requests over the bank's alternate NoC endpoints; not attempted (separate scheme change, uncertain net win).
- Accuracy achieved: bit-exact (exact gate) on every case: `int32_extremes`, the `-1` variant, uint32 high bit, bf16/fp32 RM, int32/uint32 TILE.
- Golden test progress: targeted slice (`EVAL_CAPTURE_L1=0`, `-k "ROW_MAJOR or 16x1x5120 or loose"`) 35/35 pass; the full suite is left to the harness gate.
- Issues encountered: the Tracy capture tool intermittently exits with code 3 (no CSV); re-running is enough. `G6_ABLATION=1` added to the perf-sweep harness so stubbed-kernel ablation runs record timing without the bit-exact assert.
- Tests added: `test_g6_reshape_row_major.py` knob matrix += `per_mover_read`, `no_bank_split`, `shared_8_cores` settings and two cases (`(3,10)→(5,6)` int32: whole-page atoms straddling input rows under the bank split; `(1,4095)→(1365,3)` bf16: 6-byte half-word copies); the flagged-plan pin asserts `shared_read` and `bank_split`. 100/100 under `--dev`; unit directory 173 passed.
