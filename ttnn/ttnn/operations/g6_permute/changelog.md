# Changelog: g6_permute

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation through the incremental pipeline (planner → implementer → verifier).
  - Regime `rows_tile_scatter`: each input tile's live prefix is read once into `cb_input_tiles`, and its 32 B half-rows are written straight to their output pages in DRAM. One CB, no compute kernel.
  - Input tiles are split over the full grid (`split_work_to_cores`, `row_wise=True`). `block_in_tiles` is the whole per-core assignment, capped by the 1347776 B L1 budget.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], buffer=["dram"], rank=[4], alignment=[tile_aligned, w_non_aligned, h_non_aligned], dims=[(0, 2, 1, 3)]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rms_err=0, max ULP=0 on 4 shapes (`test_g6_permute_precision_baseline.py`: (1,1,16,64), (1,6,16,64), (2,34,47,96), (1,64,100,256)).
- **Golden suite at Phase 0**: 20 / 20 SUPPORTED registry cells pass, and 12 cells are `xfail_expected`. `xpass_drift`, `xfail_wrong_mode` and `supported_fail` are all 0 (`verifier_results/verifier_report.json`). `test_regression.py` passes 4 / 7 under the graded runner; the 3 failures are a harness L1-capture artifact on host-side refusals, and they pass under `run_safe_pytest.sh`.
- **Device time**: 1.26–2.23 µs per Qwen production cell, 1 program, 2048 B L1 peak per core.
- **Issues encountered (verifier fixes)**:
  - The plain build compiled each kernel's block loop twice (a zoned copy and a tail copy). `ZONED_BLOCKS` is now 0 unless `PROFILE_KERNEL` is defined. Reader `.text` went 1264 → 840 B and writer 1952 → 1304 B on `[1,64,100,256]`; the zone build is unchanged.
  - Deleted `test_g6_permute_debug.py`. It targeted the superseded `rows_tile` regime, and its `ImportError` broke collection of the whole test directory.
  - Widened `SUPPORTED["alignment"]` with `tile_aligned` and `w_non_aligned` after 6 bit-exact shapes with poisoned padding. The regime keeps W, so padding columns stay in padding.
  - Checked and kept the writer's per-row `noc_async_write_one_packet_set_state` plus two `_with_state` half-row writes.
  - Removed dead kernel code: `RowsLive::of` and the reader's unused `tensor_at` binding.
  - The precision baseline takes max abs error from `comp_allclose` (exact tolerance).
- **Tests added**: `test_g6_permute_alignment.py`, `test_g6_permute_precision_baseline.py`. `test_g6_permute.py` (acceptance) and `test_g6_permute_scatter_debug.py` (knob coverage) were already in place.

## Refinement 1 — ROW_MAJOR rank-6 block gather (FLUX), DRAM
- Date: 2026-11-27
- What was done:
  - SUPPORTED gains layout `ROW_MAJOR_LAYOUT`, rank `6` and dims `(0, 1, 4, 2, 5, 3)`. EXCLUSIONS gains `{layout: TILE, dims: (0,1,4,2,5,3)}` and `{layout: ROW_MAJOR, dims: (0,2,1,3)}`; neither has a production case.
  - New regime `blocks_rm_gather` (`g6_permute_blocks_rm_gather.py`, `g6_permute_blocks_reader.cpp`, `g6_permute_blocks_writer.cpp`), selected on the host by `(layout, dims)`. Work unit `(n, c, h)`, split over the full grid (`split_work_to_cores`, `row_wise=True`). `block_h = min(units_per_core_g1, block_cap)`, one source of truth, capped by the 1347776 B budget; `BLOCK_H_LIMIT` and `INPUT_DEPTH` are live knobs in the cache key.
  - Reader: each 4 B input page lands in its own 64 B slot (`buffer_aligned_page_size`), so local ≡ remote mod 64 (risk R7). Every read of a block is issued before one barrier. `cb_input_slots` is double-buffered when a core owns ≥ 2 blocks.
  - Writer: gathers `[W, I·J] → [I·J, W]` in L1 with RISC-V loads and stores (bit-exact). For even `J` it moves two values per 32-bit access; odd `J` uses halfwords. It writes the `I·J` rows per unit with one `noc_async_writes_flushed` per block and one `noc_async_write_barrier` at the end.
  - `_cache_key` covers every CT input of the regime (knob key, page sizes, shape, buffer, grid). `g6_permute_plan` returns a `BlocksPlan`. The BB-05 refusal raises when one unit does not fit.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0, 0 differing bit patterns) on FLUX_73 `[1,32,64,32,2,2]`, FLUX_74 `[1,64,128,32,2,2]`, 9 other shapes of the cell (odd J and odd W, J=1/3/5/8/32/36, I=1/3/4/32, W=4…64, N=2, every alignment tag), special values (NaN payloads, ±inf, −0, denormals), negative dims, and 4 forced multi-block plans (ragged, depth 1/2/3).
- Device kernel time (Blackhole, 110 cores, DRAM interleaved, one fresh-cache run each):

  | Case | Gather | Kernel ns | NCRISC (reader) ns | BRISC (writer) ns |
  |------|--------|-----------|--------------------|-------------------|
  | L0 / FLUX_74 | halfword (first build) | 1076226 | 980165 | 1076201 |
  | L0 / FLUX_74 | word (shipped) | **1022579** | 967864 | 1022551 |
  | L1 / FLUX_73 | halfword (first build) | 255859 | 231240 | 255779 |
  | L1 / FLUX_73 | word (shipped) | **246041** | 231850 | 245938 |

  Baseline for Refinement 3: L0 = 1022579 ns, FLUX_73 = 246041 ns. L0 is below stock (1056.20 µs) by 3%.
- Finding (for Refinements 3 and 5): the read side is bound by the DRAM request rate, not by RISC issue. FLUX_74 run on 110, 55, 22 and 11 cores (`test_g6_permute_blocks_rm_debug.py`): reader time 976, 964, 1164, 1173 µs. Ten times fewer cores issuing gives the same reader time. The whole chip retires ~1.85 ns per 4 B request, ~15 ns per request per DRAM bank. Levers that only speed up one core's issue (`trid_double_issue`, `bank_stagger` alone) should not move L0. Fewer requests should: Refinement 5's bank-coalesced `k·64 B` reads. `split_reader` is worth one measurement only if the NoC1 reads reach different DRAM endpoints. The writer adds a tail after the last read: the gather of the whole block (one block per core at the design `block_h`), 96 µs halfword / 55 µs word on L0. A smaller `block_h` with depth 2 (`double_buffer`) would hide all but the last block's gather.
- Golden test progress: FLUX_73, FLUX_74, loose L0 and L1 pass (were `xfail_expected`); Qwen representatives still pass. Slice: 10 passed, 3 xfailed (Panoptic ROW_MAJOR in L1, `buffer='l1'` not yet in SUPPORTED).
- Issues encountered: the first gather used `volatile` halfword accesses, ~13 cycles per value. That made the gather tail 9% of L0, and the build was 2% slower than stock. The word gather with plain loads and stores fenced by compiler barriers cut the tail to 55 µs.
- Tests added: `test_g6_permute_blocks_rm.py` (21 cases: production shapes, 9 other cell shapes, special values, negative dims, plan report, excluded cells, program reuse, forced multi-block plans), `test_g6_permute_blocks_rm_debug.py` (grid-scaling experiment on FLUX_74, run under `--profile`).

## Refinement 2 — bfloat8_b TILE NHWC→NCHW (Panoptic p3), plus L1 placement for every regime
- Date: 2026-11-27
- What was done:
  - SUPPORTED gains dtype `bfloat8_b`, buffer `"l1"` and dims `(0, 3, 1, 2)`. EXCLUSIONS gains the four cells with no regime: `{bfloat8_b, (0,2,1,3)}`, `{bfloat8_b, (0,1,4,2,5,3)}`, `{bfloat16, TILE, (0,3,1,2)}`, `{ROW_MAJOR, (0,3,1,2)}` (the last one is Refinement 4's). The non-aligned bfloat8_b cells are not excluded: the regime handles them (below).
  - New regime `nhwc_tile_bfp8_transpose` (`g6_permute_nhwc_tile_bfp8.py`, `kernels/g6_permute_nhwc_bfp8_{reader,compute,writer}.cpp`, shared `g6_permute_nhwc_bfp8_common.hpp`), the op's first compute kernel, selected on the host by `(TILE, (0,3,1,2))`. Work unit = group `(n, ht, wt, ct)`, split over the full grid (`split_units`, `row_wise=True`). Per group:
    1. reader: 32 input tiles `(n, h, wt, ct)` into `cb_input_tiles` (Bfp8_b), pushed in chunks of `READ_CHUNK_TILES = 8`, one barrier per chunk;
    2. compute: `transpose_block` into DEST, 8 tiles per batch (cumulative wait, so each batch starts when its slots land), `pack_untilize_dest` in bfloat16 to `cb_row_block`, page `h` = row `h` of a `[32, 32·32]` block;
    3. compute: the tilize helper (32 tiles wide) builds output tile `c` from column chunk `c` of every row and packs bfloat8_b; the packer sets each 16-value group's exponent to its largest value. This tilize is the design's core-local row exchange, so no dataflow copies exist;
    4. writer: the `c_live` output tiles, one flush per group, one write barrier at the end.
  - Raw LLK only for step 2 (`transpose_block` + `pack_untilize_dest`): kernel_lib has no transpose helper, and the untilize helper's unpack init has no transpose. Recorded at the kernel head.
  - Non-aligned shapes: H not a multiple of 32 leaves the trailing slots unread; they reach only output padding rows, each a bfloat8_b group of its own. W not a multiple of 32 puts input padding rows into live output groups, so the reader zeroes their mantissas (`noc.async_write_zeros`); a zero mantissa never raises a shared exponent. C not a multiple of 32 skips the empty output tiles.
  - `group_depth = min(GROUP_DEPTH = 2, groups per core)`, lowered when two groups exceed the budget. `RuntimeError` (BB-05) when one group plus the output's L1 pages exceeds 1347776 B. `knob_key` covers `GROUP_DEPTH` and `READ_CHUNK_TILES`. `g6_permute_plan` returns an `NhwcPlan`.
  - Buffer `"l1"` on `rows_tile_scatter` and `blocks_rm_gather`: a knob-turn, as designed. The accessors address L1 banks, and `block_cap` subtracts the output's own L1 pages per core.
- Accuracy achieved: Panoptic p3 `[1,128,256,32]` randn, L1 and DRAM: PCC=0.999989, relative RMS=4.73e-3 against the permuted bfloat8_b input; PCC=0.999978, relative RMS=6.72e-3 against the host bfloat8_b rendering, 95.4 % of values identical to it. The device packer rounds differently from the host, which the one-step gate accepts. Every value is within one step `2^(E−133)` on p3 (randn, `w_scaled`, `bf8_odd_grid` with the shared-exponent check) and on 7 other shapes: H tail, C tail, W tail with poisoned padding (1e4, inf, −3e4), N = 2, 256 groups. A group of tiny values (`randn · 2^-124`, bfloat16 denormals after unpack) also passes.
- Device kernel time, Panoptic p3 (Blackhole, 32 active cores, one group per core, one fresh-cache run each):

  | Placement | Whole-group push | `READ_CHUNK_TILES = 8` (shipped) | Change |
  |-----------|------------------|----------------------------------|--------|
  | L1 (production) | 8942 ns | **8157 ns** | −8.8 % |
  | DRAM | 11494 ns | **10259 ns** | −10.7 % |

  Zones with the whole-group push (L1 / DRAM): read 1.4 / 3.1 µs, then compute 5.5 µs, then write 1.1 µs, all serial. The chunked push overlaps most of the read with the transposes.
- Finding (for the perf rounds): p3 is compute-bound on 32 of 110 cores. A group is 32 input tiles, and p3 has only 32 groups. Each core runs 64 tile passes (32 transpose + untilize, then a 32-tile tilize + bfloat8_b pack), about 116 cycles per tile. Two levers were left untried here because each changes the scheme:
  - Split a group into its two face-row halves. One core takes input tiles `h` 0–15, another `h` 16–31, so p3 runs on 64 cores. Each half writes faces 0/1 or faces 2/3 of every output tile (32 B exponents + 512 B mantissas, 16 B-aligned). This halves the transposes per core. The tilize still covers 32 tiles, unless the helper's `total_input_pages` row count makes the half-empty tiles cheaper.
  - Let the writer start before the whole group is tilized. Today it waits for all 32 output tiles, a 1.1 µs tail. The tilize helper needs contiguous rows, so a sub-block tilize needs `cb_row_block` split by `c` chunk.
- Golden test progress: slice of 17 (PANOPTIC_53, loose L2, L15, L16, L18, FLUX_73, FLUX_74, L0, L1, Qwen 303 and its loose variants): 14 passed, 3 xfailed. The 3 xfails are the ROW_MAJOR Panoptic cells (`PANOPTIC_44`, L3, L17), excluded until Refinement 4. Expected full suite: 29 / 32 passing, 3 `xfail_expected`. `test_regression.py`: 7 / 7.
- Issues encountered: this regime's first attempt was interrupted before its commit. Its uncommitted kernels and tests were reviewed, re-run and kept. `run_safe_pytest.sh` and `tt-probe.sh` still need `LD_PRELOAD=<mpix_stub>` on this host (ULFM OpenMPI is missing). Under `--profile`, a `-k` filter with spaces is split into separate arguments; pass node ids instead.
- Tests added: `test_g6_permute_nhwc_bfp8.py` (31 cases):
  - Panoptic p3 in L1 and in DRAM, its adversarial groups, and a precision record;
  - 7 other shapes of the cell, W-poisoned odd-grid shared exponent, and a tiny-exponent group;
  - negative dims, the plan report, the excluded cells, and program reuse;
  - forced `GROUP_DEPTH` 1/3 and forced `READ_CHUNK_TILES` 1/32;
  - Qwen 303, `[2,34,47,96]`, FLUX_73 and an odd-J FLUX shape in L1, bit-exact.
