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
