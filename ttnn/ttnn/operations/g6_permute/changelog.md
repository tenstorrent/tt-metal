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

## Refinement 3 — Speed up the perf-flagged FLUX profile (gather tuning)
- Date: 2026-11-27
- What was done:
  - Reused: `blocks_rm_gather`'s reader/writer pair, plan, CBs, cache key and plan report. Added: four read-side knobs on the host (`g6_permute_blocks_rm_gather.py`, all in `knob_key()` and reported by `BlocksPlan`), plus the reader paths they select (`g6_permute_blocks_reader.cpp`). The writer is unchanged.
    - `BLOCK_PAGES = 256`: `block_h = ceil(BLOCK_PAGES / (W·I))` under the L1 and core caps. This is the `double_buffer` lever: FLUX_74 goes from one 75-unit block per core to blocks of 4 units, and the gather of every block but the last runs under the reads. `None` restores the whole-assignment block.
    - `READ_ALT_NOC_SHARE = 1/2`, DRAM input only: half of each bank's page rounds go on NoC1. On Blackhole each NoC enters a DRAM bank through its own port. Both kernels run in dynamic NoC mode when the share is nonzero.
    - `BANK_MAJOR = True`, DRAM input only: the block is issued bank by bank, one `noc_async_read_one_packet_set_state` per (bank, NoC) run and `_with_state` reads inside it (the refinement's bank-major walk).
    - `BANK_RUN_ROUNDS = 8`: a bank-major chunk is 8 bank rounds, which bounds a core's run of requests to one port.
    - `BANK_STAGGER` (`bank_stagger`) stays a live knob at its trivial default `False`. It measured flat.
  - The interrupted earlier attempt's `READ_NOCS` 1/2/3 knob became `READ_ALT_NOC_SHARE`. The adaptive `cmd_buf_ready` mode (3) was dropped: it measured no better than the fixed half split.
- Bottleneck classification (FLUX_74, Blackhole, 110 cores, one fresh-cache run per variant):
  - **Not RISC-issue-bound on DRAM.** On 55 cores the kernel is 594966 ns against 551599 on 110 (+8 %). Each reader has at least 2× issue slack.
  - **Not writer-bound.** Ablation of the shipped plan: full 551296 ns, writes stubbed 542144 (−1.7 %, noise), reads stubbed 64985, everything stubbed 49847 (the gather plus CB handshakes).
  - **Bound by the per-port DRAM request rate.** One NoC means 8 ports: ~14.8 ns per 4 B read per port. Two NoCs mean 16 ports: ~17 ns per read per port. An uneven split loses on both sides: share 15/32 gives 628709 ns, 17/32 gives 628919, 1/2 gives 603150.
  - **L1 input is issue-bound.** The R1 reader takes ~28 ns per read: 132 µs for 4766 reads per core.
- Lever measurements (FLUX_74 DRAM, device kernel ns):

  | Lever | Setting | Kernel ns |
  |-------|---------|-----------|
  | R1 plan | one 75-unit block, page order, NoC0 | 1019021 – 1034835 |
  | `double_buffer` | block 2 units, depth 4 | 939380 |
  | `double_buffer` | block 1, depth 4 | 943812 |
  | `bank_stagger` | block 2, depth 4 | 947080 |
  | 2-NoC split | share 1/2, block 2, depth 4 | 602124 |
  | 2-NoC split | share 7/16 / 15/32 / 17/32 | 655101 / 628709 / 628919 |
  | 2-NoC + page order | block 1 / 4 | 579520 / 609748 |
  | bank-major, NoC0 only | block 2 | 1090015 |
  | bank-major + 2-NoC | run = whole block: block 1 / 2 / 4 / 8 / 75 | 556944 / 567919 / 627646 / 701654 / 933225 |
  | bank-major + 2-NoC | run 2 / 4 / 8 rounds, block 1 | 579761 / 565899 / 550793 |
  | bank-major + 2-NoC, run 8 | block 2 / 4 / 8 | 550575 / 558580 / 564223 |
  | bank-major + 2-NoC + stagger | block 2 | 570460 |
  | input depth | 2 / 4 / 8 at block 1 | 554308 / 551599 / 550061 |
  | fewer cores | 88 / 77 / 66 / 55 / 44 | 566093 / 561293 / 562584 / 594966 / 596410 |
  | `BLOCK_PAGES` (shipped plan) | 64 / 128 / 256 / 512 | 553389 / 550625 / 552775 / 560955 |
  | `BLOCK_PAGES` (shipped plan), L1 input | 64 / 128 / 256 / 512 | 152461 / 143413 / 141308 / 139160 |

  On L1 input the 2-NoC split (244256) and the bank-major walk (259813) both lose to page order on NoC0 (152215). Both are therefore gated to DRAM input.
- Device kernel time, guard set (final run, same session):

  | Cell | R1 plan | Shipped | Change |
  |------|---------|---------|--------|
  | L0 / FLUX_74 DRAM | 1029265 | **554900** | −46.1 % (−45.7 % against the recorded 1022579) |
  | FLUX_73 DRAM | 248450 | **144460** | −41.9 % |
  | FLUX_74 L1 | 187733 | **141290** | −24.7 % |
  | FLUX_73 L1 | 48597 | **38410** | −21.0 % |

  L0 is now 0.53× stock (1056.20 µs). `rows_tile_scatter` and `nhwc_tile_bfp8_transpose` kernels are untouched.
- Finding (remaining headroom):
  - L0 is bound by the DRAM ports' request rate: 524288 single-page reads over 16 ports at ~17 ns each. Only fewer requests move it, i.e. Refinement 5's bank-coalesced `k·64 B` reads.
  - Not built:
    - `split_reader` and `trid_double_issue`: the core-count, depth and block sweeps show the reader never starves the ports on DRAM.
    - The third NoC endpoint of each DRAM bank. The soc descriptor ties the noc0 endpoint to the one CMFW uses for telemetry (SYS-1419), so workers stay on the two endpoints the descriptor assigns.
  - For L1 placement the reader is issue-bound, so a split reader or a cheaper per-page address (bank cursor instead of a `TensorAccessor` lookup per page) is the next lever there.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0, no differing bit patterns) on FLUX_73, FLUX_74, every `test_g6_permute_blocks_rm.py` shape, 6 new forced read-path plans, and every sweep variant above, in DRAM and L1.
- Golden test progress: FLUX slice 4 / 4 pass (FLUX_73, FLUX_74, loose L0, L1). The other regimes' code is untouched; expected full suite unchanged at 29 / 32 passing, 3 `xfail_expected` (Refinement 4's ROW_MAJOR Panoptic cells).
- Issues encountered: one Tracy capture crashed (`tracy-capture` exit 3) and its CSV was never written; re-run. One 18-op profiled run dropped its last 4 ops from the CSV; those variants were re-measured in smaller runs.
- Tests added:
  - `test_g6_permute_blocks_rm.py`: `test_forced_read_paths_bit_exact` covers 6 plans: bank-major chunks ending mid-round, stagger, shares 1/4 and 3/8, page order with the 2-NoC split, and bank-major on one NoC. The plan test now checks the `BLOCK_PAGES` block and the DRAM read plan. The forced-block fixture pins `BLOCK_PAGES = None` so its ragged multi-block plans stay ragged.
  - `test_g6_permute_blocks_rm_perf.py`: the guard sweep, shipped plan against R1 on FLUX_73/74 in DRAM and L1.
  - `test_g6_permute_blocks_rm_ablation_debug.py`: ablation arms behind temporary kernel guards, plus the `BLOCK_PAGES` sweep.
  - `test_g6_permute_blocks_rm_debug.py`: the grid-scaling experiment now runs at share 0 and 1/2.

## Refinement 4 — ROW_MAJOR NHWC→NCHW (Panoptic p4)
- Date: 2026-11-27
- What was done:
  - EXCLUSIONS loses `{layout: ROW_MAJOR, dims: (0,3,1,2)}`. SUPPORTED is unchanged: ROW_MAJOR, `"l1"` and `(0,3,1,2)` were already in it.
  - New regime `nhwc_rm_transpose` (`g6_permute_nhwc_rm_transpose.py`, `kernels/g6_permute_nhwc_rm_writer.cpp`), selected on the host by `(ROW_MAJOR, (0,3,1,2))`, DRAM or L1.
    - Work unit `(n, h, wb)`: `block_w` consecutive input pages `(n, h, w)` of `C` values. The last unit of each row holds `tail_w = W − (⌈W/block_w⌉ − 1)·block_w` pages. Units are split over the full grid (`split_units`, `row_wise=True`).
    - `block_w` is `W` (whole rows) or a multiple of 8 values, so each partial-row write at byte offset `wb·block_w·2` stays 16 B-aligned like its staging row. `BLOCK_W` is a live knob; `None` lets `_choose_block_w` pick the width with the lowest modelled cost on the busiest core, among widths whose unit fits the budget.
    - A block is `block_units = ⌈BLOCK_PAGES / block_w⌉` units (`BLOCK_PAGES = 256`) under the L1 and core caps. `cb_input_slots` is double-buffered when a core owns ≥ 2 blocks.
    - Writer: transposes `[width, C] → [C, width]` in L1 (word path for even `C`, halfwords for odd `C`) and writes the `C` partial rows. One flush per block, one write barrier at the end.
    - `RuntimeError` (BB-05) when no unit width fits one unit plus the output's L1 pages in 1347776 B. `knob_key` covers every knob. `g6_permute_plan` returns an `NhwcRmPlan` that reports `block_w`, `tail_w`, `block_units`, `l1_bytes_per_core` and `out_l1_bytes_per_core`.
  - Reused:
    - `blocks_rm_gather`'s reader. It now takes rows of units with a shorter last unit (`units_per_row`, `tail_pages` CT args, a `row_pos` RT arg) and a start page instead of a start unit. FLUX passes one unit per row, which compiles to the old block page count. Its four DRAM read paths (2-NoC split, bank-major runs, stagger) serve both regimes.
    - `read_plan`, `reader_descriptor` and `dm_config`, factored out of `g6_permute_blocks_rm_gather.py`.
    - The transpose itself, moved to `kernels/g6_permute_rm_transpose.hpp` (`transpose_pages`) and called by both writers. For FLUX (`J = 2`) it issues the same loads and stores in the same order as before.
  - Added: the host module, the NHWC writer, the shared header, tests.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0, 0 differing bit patterns) on:
  - Panoptic p4 `[1,128,256,32]` in L1 and in DRAM;
  - 10 other shapes of the cell: odd C (3, 5), C = 1, C = 64 (128 B pages), W = 5 (whole rows below one write granule), tails of 1, 4, 8 and 16 values, W = 1000 cut over 63 cores, N = 2, every alignment tag;
  - special values (NaN payloads, ±inf, −0, denormals) and negative dims `(0, -1, 1, -2)`;
  - 5 forced plans: `block_w` 8/16/24/32/whole rows, multi-block rings across rows, depth 1/2/3, and in DRAM page order on one NoC and bank-major with a 1/4 NoC1 share.
- Device kernel time, Panoptic p4 (Blackhole, 110 cores, one fresh-cache run per width, `test_g6_permute_nhwc_rm_perf.py`):

  | `block_w` | units / core | L1 (production) ns | DRAM ns |
  |-----------|--------------|--------------------|---------|
  | **64 (chosen)** | 5 | **59226** | **87339** |
  | 256 (whole rows) | 2 | 83693 | 106596 |
  | 128 | 3 | 75144 | 99361 |
  | 32 | 10 | 81181 | 109136 |
  | 16 | 19 | 112749 | 141174 |

  `_choose_block_w` picks the measured optimum in both placements. Its weights are relative and calibrated to put the model's minimum on this sweep; they are not measured per-operation costs.
- Regression check: FLUX_74 / L0 after the shared-transpose refactor is 555877 ns, against Refinement 3's 554900 (+0.2 %, noise).
- Finding (for the perf rounds): p4 is bound by the writer RISC. In L1, BRISC takes 59.1 µs and the reader (NCRISC) 10.1 µs, so the reader idles ~83 % of the kernel.
  - From 32 to 64 values per unit, the writer time drops 22 µs for 160 fewer writes per core. That puts each partial-row write at ~137 ns. Per transposed value, the cost changes with the unit width, so a linear model does not fit the sweep.
  - Levers left untried, because they change the kernel roles or are perf work outside this refinement:
    1. Split each core's units between both RISCs, each reading, transposing and writing its own half on its own NoC (catalog `split_reader`).
    2. Make write addresses cheaper: advance a bank cursor by `H` per channel instead of one `TensorAccessor` lookup (a division by 110 banks) per partial row.
    3. Try a transpose store order that walks rows sequentially.
- Golden test progress: slice of 5 (PANOPTIC_44, loose L3 and L17, plus FLUX_73 / L1 and L18 as prior-passing representatives): 5 / 5 pass. PANOPTIC_44, L3 and L17 were `xfail_expected`. Expected full suite: 32 / 32 passing, 0 `xfail_expected`.
- Issues encountered:
  - The pre-commit black hook (23.10.1) failed on every file with "Ran out of input". Its cache file under `~/.cache/black/23.10.1/` was empty; removing it fixed the hook.
  - `LD_PRELOAD` of the MPI stub is still needed on this host.
- Tests added:
  - `test_g6_permute_nhwc_rm.py` (21 cases): p4 in L1 and DRAM, 10 other shapes, special values, negative dims, the plan report (including the 38 × 512 B output pages per core), program reuse, and 5 forced plans.
  - `test_g6_permute_nhwc_rm_perf.py`: the `BLOCK_W` sweep, run under `--profile`.
  - Unit regression net still green: `test_g6_permute_blocks_rm.py`, `test_g6_permute_nhwc_bfp8.py` (incl. FLUX in L1) and `test_g6_permute.py`, 82 / 82.

## Refinement 5 — FLUX profile: bank-coalesced reads
- Date: 2026-11-27
- What was done:
  - Regime `blocks_rm_bank_coalesced` is built and selected by measured result. `READ_MODE = "auto"` picks it for DRAM and L1 input (`COALESCED_BUFFERS`); `g6_permute_plan` reports `regime`, `coalesced`, the staging and the write-group plan. No SUPPORTED change.
  - Reused: `blocks_rm_gather`'s reader, writer, plan, split and cache key. The interrupted earlier attempt of this refinement left the coalesced reader in the tree; it was reviewed, measured, kept and extended. Added, all host knobs in `knob_key()`:
    - Reader (`coalesced` CT branch): each bank's pages of a block arrive in one read of whole slots (two when split over the NoCs) into the private `cb_bank_staging`. The reader then copies the payloads into `cb_input_slots` in page order at a 4 B stride. `COALESCED_BLOCK_PAGES` is per placement, 256 on DRAM and 2048 in L1; `STAGING_DEPTH = 1`.
    - `COMPACT_BATCH_WORDS = 8` and `GATHER_PAIR_BATCH = 8`: the staging copy and the writer's gather issue a batch of loads before their stores (`std::integer_sequence`; DM kernels build at -Os, which does not unroll). FLUX_74 per block: copy 4356 → 1030 cycles, gather 3558 → 2144. `transpose_pages` gains a `pair_batch` template parameter, default 1, which the NHWC writer keeps.
    - Writer twin, grouped writes (`WRITE_GROUP_UNITS = 64`, `WRITE_GROUP_DEPTH = 2`, DRAM output with `J | num_banks`). For a fixed `(n, c, i)`, consecutive `h` give consecutive output pages, so one bank's pages are consecutive slots. Up to 64 units of one `(n, c)` form a group. The gather places their rows bank by bank in `cb_output_rows`; the J rows of a `(unit, i)` sit one bank region apart and never wrap. Each (run `i`, bank) is then one write of whole slots. Two group buffers alternate, each with its own write transaction id. FLUX_74 goes from 300 × 64 B writes per core to 32–48 writes of ≤ 1088 B.
- Device kernel time (Blackhole, 110 cores, one fresh-cache run each; final guard run in one session):

  | Cell | R3 recorded | R3 gather, this session | Shipped | Change vs R3 |
  |------|-------------|-------------------------|---------|--------------|
  | L0 / FLUX_74 DRAM | 554900 | 545233 | **83545** | −84.9 % (6.6×) |
  | FLUX_73 DRAM | 144460 | 142244 | **25486** | −82.4 % |
  | FLUX_74 L1 | 141290 | 140964 | **67718** | −52.1 % |
  | FLUX_73 L1 | 38410 | 37907 | **25233** | −34.3 % |
  | Panoptic p4 L1 / DRAM (shares the reader and transpose) | 59226 / 87339 (R4) | — | 59219 / 87228 | unchanged |

  `rows_tile_scatter` and `nhwc_tile_bfp8_transpose` are untouched.
- Lever measurements (FLUX_74 DRAM unless noted, device kernel ns, ±3–5 % between sessions):

  | Step | Setting | Kernel ns |
  |------|---------|-----------|
  | inherited coalesced reader | page-by-page copies, 2 stages, block 256 | 95405 – 100889 |
  | | block 128 / 512 / 1024 | 101810 / 128100 / 123051 |
  | batched copy only | 2 stages | 121770 – 122975 |
  | | 2 stages + stagger / NoC0 only / block 128 | 123144 / 117638 / 110825 |
  | | 1 stage | 91903 – 94657 |
  | batched gather only | 2 stages, 4 / 8 pairs | 97344 / 98136 |
  | batched copy + gather | 2 stages | 115881 – 116208 |
  | | 1 stage | **89253** |
  | + grouped writes | 8 / 16 / 32 / 48 / 64 / 128 units | 89849 / 86284 / 84681 / 83512 / **82900–83743** / 82156 |
  | | 32 units, one buffer (two: 84681) | 86357 |
  | | 64 units with 2 stages / block 512 / block 128 / NoC0 only | 83847 / 86291 / 93561 / 96781 |
  | L1 input, coalesced | block 256 / 512 / 1024 / **2048** / 4096 | 188487 / 126726 / 72297 / **67720** / 91942 |
  | | block 1024 / 2048 with 2 stages | 73379 / 73414 |
  | | grouped writes into L1 output | 78067 (FLUX_73: 26623 against 25239) |

- Bottleneck classification (ablation, every payload stubbed behind temporary guards, CB handshakes kept):
  - Inherited reader (95.4 µs). Reads alone took 70.9 µs, the copies alone 69.2 µs, the writes alone 20.1 µs, and the sync floor was 4.9 µs. Zones showed the reader at 4355 cycles of copying per 256-page block against 443 cycles waiting on DRAM, so the RISC copies were binding.
  - Faster copies with two stages make it slower. The reader then keeps reads in flight without a break, and the cores at x = 13–15 end 20–40 µs late. Their write issue rises 3300–4600 cycles per block against 1336 elsewhere, and their read barrier 2000+ cycles against 350: NoC congestion on the links into the far columns. One stage lets each core's reads pause during its copy, and the late columns disappear.
  - Shipped plan (83.3 µs). Reads alone take 71.2 µs, 32 MiB at ~508 GB/s, the DRAM peak. Writes add 5.4 µs (11.5 µs before grouping), the copies 5.4 µs, sync floor 5.2 µs. The read bytes bound it now: every 4 B payload sits in its own 64 B DRAM slot.
- Findings (remaining headroom, not filed as follow-ups):
  - The `read_coalescing` bank-affinity arm does not apply. A unit's 64 input pages span all 8 banks, so a core reading one bank would need the other banks' pages of its units from other cores. Rotating each core's bank order (`COALESCED_BANK_STAGGER`) measured flat; it stays a live knob at `False`.
  - Next levers:
    1. Fuse the staging copy into the gather, reading payloads straight from the bank-ordered slots, to drop one RISC pass of 4800 words per core.
    2. Cap the reads a core keeps in flight, so two stages stop congesting.
  - Not tried: each changes the reader–writer contract, and the read floor leaves at most ~12 µs.
  - L1 input stays on one NoC: the 2-NoC split loses there (Refinement 3).
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0, 0 differing bit patterns) on:
  - FLUX_73 and FLUX_74 in DRAM and L1, and every sweep variant above (each checks bit for bit);
  - every `test_g6_permute_blocks_rm.py` shape;
  - 8 forced coalesced plans: blocks starting mid bank round, ragged blocks across `(n, c)`, fewer pages than banks, 2 B and 72 B pages, NoC shares 0 / 1/4 / 3/8, stagger, page-by-page and odd batches;
  - coalesced L1 input on FLUX_73 and an odd-J shape;
  - 7 forced write-group plans: groups cut by the core range and by `(n, c)`, J = 1 / 2 / 4 / 8, 128 B output slots (W = 33), one buffer, the gather read path, page-by-page copies;
  - the `J ∤ num_banks` fallback.
- Golden test progress: ROW_MAJOR slice 7 / 7 passed (FLUX_73, FLUX_74, loose L0 and L1; PANOPTIC_44, loose L3 and L17 on the shared reader and transpose). The other regimes' code is untouched; expected full suite unchanged at 32 / 32.
- Issues encountered:
  - The interrupted attempt left `test_plan_reports_regime` asserting `blocks_rm_gather` on DRAM; it now checks the coalesced plan, and a new test checks the gather plan. The forced gather tests pin `READ_MODE = "gather"` so they keep covering that path.
  - `tracy-capture` crashed (exit 3) on most profiled runs, so no ops CSV was written. The device log is complete, so kernel time was taken from it: the span of the `*-KERNEL` zones of each program. It matched the CSV exactly on the runs that produced one.
- Tests added:
  - `test_g6_permute_blocks_rm.py`: `test_plan_reports_gather_regime`, `test_forced_coalesced_bit_exact` (8), `test_coalesced_in_l1_bit_exact` (2), `test_forced_write_groups_bit_exact` (7), `test_write_groups_need_j_dividing_banks`.
  - `test_g6_permute_blocks_rm_perf.py`: coalesced, batch, staging, write-group and L1 block variants.
  - `test_g6_permute_blocks_rm_ablation_debug.py`: copy and gather ablation arms.

## Perf 1
- Date: 2026-11-27. Blackhole p150b (chip 3 of bh-lb-18), 110 worker cores, 1350 MHz. Every time is `DEVICE KERNEL DURATION [ns]`, the median of launches 2–12 of the case bench (`generated/meta/g6_permute/bench/bench_cases.py`), profiled plain (no sum zones) unless noted. `SUPPORTED` is unchanged.
- Goal (operator): no production case (LOOSE_CASES 0–11) slower than stock `ttnn.permute`. The ideas went to Panoptic p44 (A, 3.46× slower than stock) and the Qwen cases (B). FLUX_74, the flagged case, already beat stock 12.5× and stayed a guard cell. It was not re-ablated this round; Refinement 5 left it bound by its DRAM reads (32 MiB of 64 B slots).

### Measured breakdown (p44, `nhwc_rm_transpose`, before this round)
- Zones per core (BRISC median, cycles): RISC-V transpose 61.7k (45.7 µs), wait for the first 256-page block 10.0k (7.4 µs), write issue 6.8k for 160 writes of 128 B (42 cycles a write). Reader (NCRISC): issue 12.0k for ~298 pages (~37 cycles a page), barrier 0.5k.
- Cumulative ablation (payloads stubbed, CB handshakes kept; `tests/.../perf1/test_ablation_p44.py`):

  | Arm | ns |
  |---|---|
  | full | 59,253 |
  | − transpose | 13,499 |
  | − transpose − writes | 10,223 |
  | − transpose − writes − reads (sync floor) | 957 |
  | − reads only | 51,971 |
  | − writes only | 55,007 |

- Ranked: transpose 45.8 µs (77 %), reads 9.3 µs (issue-bound, not NoC-bound: the L1 floor of p44 is 3.99 µs), writes 3.3 µs.
- Disagreement with the operator's inference: the writes cost ~3.3 µs (42 cycles each), not ~22 µs at ~137 ns. The transpose is 45.8 µs, not ~37 µs.
- Qwen (B): the operator's start-skew diagnosis held. `first2last` was 390–463 ns row-first on 12–32 cores, and 87–213 ns column-first.

### Portfolio and verdicts
| Idea | Verdict | p44 L1 before → after (ns) | Domain / notes |
|---|---|---|---|
| operator candidate 1: column-first core fill | **WIN, graduated** | Qwen, see the guard table | Applies to every regime. Carve-out `nhwc_tile_bfp8_transpose`: column-first measured p53 8,179 → 10,637 (+30 %). FLUX and p44 flat. |
| `tensix_transpose` (operator candidate 2): tilize → transpose → pack-untilize on the Tensix engine | **WIN, graduated** | 59,244 → 11,613 (subagent); 11,820 in the op | Bit-exact only with **UInt16** CBs. Float16_b broke 509 of 65,536 bf16 patterns: denormals → 0, NaN payloads lost (0xff98 → 0xff80), −0 → +0. Correct for every shape tried (C 1–128, W tails, whole rows, DRAM/L1). |
| `tensix_transpose` variant `fused_d2b` (one pass, raw MOVD2B/TRNSPSRCB) | not graduated | 11,613 vs 11,687 for `two_pass` | End to end −0.6 %, inside noise; it only wins compute-only (1.9 vs 3.4 µs). Raw LLK, Blackhole-only. |
| `riscv_transpose`: faster RISC-V code + transpose on the three TRISCs + read split | WIN, superseded | 59,247 → 12,425 | Conflicts with `tensix_transpose` on the same stage and loses to it. Its sync uses CB stream registers as counters. Measured menu: pair_batch 8 → 48.3 µs; new store order 34.6 µs; DM-only split 24.5 µs; TRISCs 14.6 µs. |
| `rm_issue_cost`: bank-cursor read/write issue, unit pushes, trid pipeline, DRAM coalesced reads | **WIN, partly graduated** | full op 59,249 → 51,587; transpose-stubbed 13,507 → 7,379 | Read issue 37.5 → 22.7 cycles a page (18.2 with two command buffers); write issue 42.6 → 27 (21.5). Unit pushes and the trid pipeline were already in `tensix_transpose`'s reader. Graduated: single-buffer cursor reads (L1) and writes. Not integrated: the two-buffer form, and DRAM coalesced reads (measured only on the old design, DRAM full op 87.1 → 60.8 µs; no production case of this regime is in DRAM). |
| standing idea: read once, multicast to the group | not applicable | — | No core re-reads an operand: each input page is read by exactly one core. |

### What graduated
1. **Column-first fill** (`split_units(row_wise=False)` default; commit "fill cores column-first ..."). Kernels unchanged.
2. **Tensix transpose** (commit "transpose nhwc_rm_transpose on the Tensix engine"). The RISC-V transpose writer is deleted; `g6_permute_rm_transpose.hpp` now serves only FLUX's writer.
   - Domain: every `nhwc_rm_transpose` shape. Unit width is a multiple of 8 values, at most 8 w-tiles, or the whole row. `_choose_block_w` models busiest-core time plus one unit's fill/drain; the max-only model put C=128 on whole rows (17.7 µs against 14.4 µs).
   - Reader by placement, measured both ways:
     - L1: own page-order reader, one push per w-tile on a transaction-id pipeline. The shared reader measured 13.0 µs against 11.6, and its slots are not tilize rows when C % 32 ≠ 0 in L1.
     - DRAM: the shared bank-major / 2-NoC reader, one unit per block. The page-order reader measured 63.9 µs against 41.9.
3. **Cursor issue** (commit "issue nhwc_rm_transpose reads and writes from a bank cursor"). The cursor is the default, with two measured carve-outs that keep `noc_async_read/write` (via `InterleavedAddrGen`):
   - Pages and rows above 128 B: at the cursor's rate 110 cores demand ~1.2 KB/cycle and delivered NoC throughput collapses. C=128 L1: 14,413 → 18,846 (+31 %); the reader's first unit went from 5.0k to 9.6k cycles. Marginal pairs on C=128: cursor reads alone 17,836, cursor writes alone 15,006, both plain 13,676.
   - DRAM-output writes: cursor writes crowd the reader's requests at the DRAM ports. [1,256,512,32] DRAM: 185.7k / 186.3k plain against 192.5k / 193.0k cursor, alternating runs (+3.6 %). p44 DRAM −0.6 % and C=64 DRAM −2.5 % were flat.
- Instrumentation:
  - Each RISC has a scoped zone around its first unit (the fill) plus sum zones over every unit: `PERM-RD-ISSUE/BARRIER`, `PERM-CP-TILIZE/TRANSPOSE`, `PERM-WR-WAIT/ISSUE`.
  - Sum zones record only under `TT_METAL_PROFILER_SUM=1` and compile out otherwise. Scoped zones per w-tile cost 767 ns on p44 (12,439 vs 11,672); the new set costs 148 ns.
- p44 L1 critical path after the Tensix transpose (sum zones, cycles per core): reader issue 12.1k of the reader's 13.8k (~38 cycles a page); writer issue 6.8k + wait 8.5k. The cursor then took the kernel from 11,820 to 8,006.
- Domain sweep (`tests/.../perf1/test_perf1_shapes.py`, ns). The subagent measured the old op on the same shapes; p44 is this round's own baseline.

  | Shape | Old op | Tensix | + cursor (shipped) |
  |---|---|---|---|
  | p44 L1 | 59,243 | 11,824 | **8,007** |
  | p44 DRAM | 87,164 | 43,673 | 43,396 |
  | [1,64,256,64] L1 | 60,979 | 10,410 | 7,394 |
  | [1,64,256,64] DRAM | 79,162 | 29,250 | 28,095 |
  | [1,64,256,128] L1 | 120,193 | 14,421 | 13,567 |
  | [1,128,256,24] L1 | 46,884 | 11,603 | 7,898 |
  | [1,128,256,3] L1 | 17,267 | 10,971 | 7,516 |
  | [1,8,1000,32] L1 | 15,904 | 6,213 | 4,395 |
  | [1,256,100,32] L1 | 50,153 | 11,559 | 8,598 |
  | [1,256,512,32] DRAM | 277,981 | 161k–186k (two sessions) | 191k |
  | [2,7,5,32] L1 | 3,174 | 2,335 | 1,861 |
  | [1,3,100,3] DRAM | 2,300 | 2,294 | 2,260 |

  [1,256,512,32] DRAM is session-noisy: the same Tensix commit measured 161,146 and 185,516. Against the second session it ends +3 %, the cost of the same DRAM-bound reader.

### Guard set: every production case, same session (ns; stock = `ttnn.permute` through the same bench)
| Case | Stock | Before | After | After vs stock |
|---|---|---|---|---|
| f74 FLUX_74 | 1,040,312 | 82,584 | 82,899 | 12.5× faster |
| f73 FLUX_73 | 267,122 | 25,557 | 25,552 | 10.5× |
| p53 PANOPTIC_53 | 39,326 (3 programs) | 8,160 | 8,150 | 4.8× |
| **p44 PANOPTIC_44** | 17,135 | 59,243 | **8,015** | **2.14×** (was 0.29×) |
| q301 | 2,217 | 1,801 | 1,814 | 1.22× |
| q302 | 1,418 | 1,367 | 1,133 | 1.25× |
| q303 | 2,317 | 2,184 | 1,892 | 1.22× |
| q304 | 1,718 | 1,670 | 1,423 | 1.21× |
| q305 | 1,632 | 1,333 | 1,338 | 1.22× |
| q306 | 1,269 | 1,306 | 1,027 | 1.24× |
| q307 | 1,729 | 1,704 | 1,411 | 1.23× |
| q308 | 1,627 | 1,609 | 1,313 | 1.24× |

- No case is slower than stock. FLUX, p53, q301 and q305 are unchanged within noise.
- Golden: 39 / 39 pass (`eval/golden_tests/g6_permute/`). Unit suites: 132 / 132. The nhwc suite also passes under `--dev`, with a new test that moves all 65,536 bf16 patterns in L1 and DRAM, and new forced plans for 8 w-tiles, a partial last w-tile and one rows page.

### Operator candidates
- operator candidate 1: graduated — column-first fill in every regime but `nhwc_tile_bfp8_transpose` (p53 +30 % column-first, measured). Before → after: q306 1,306 → 1,027; q308 1,609 → 1,313; q307 1,704 → 1,411; q302 1,367 → 1,133; q304 1,670 → 1,423; q303 2,184 → 1,892; q301 1,801 → 1,814 and q305 1,333 → 1,338 (2 cores, flat).
- operator candidate 2: graduated — Tensix transpose on UInt16 CBs (Float16_b is not bit-exact). p44 59,243 → 11,820, then 8,006 with the cursor issue.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` | capability | Waits for and pops its input one tile-row block at a time, and cannot tilize a tile-row at an offset inside a CB page that holds several. The DRAM reader pushes one page per unit of kw w-tiles, so w-tile wt must be addressed at `wt * ct` from the read pointer (`tilize_block` with an input offset). | n/a (not expressible with the unit-page CB) | 11,820 (p44, whole op) | `g6_permute_nhwc_rm_compute.cpp:49` |
| no helper covers this (transpose family) | capability | kernel_lib has no transpose helper, and its untilize helper has no transposing unpack, so transpose-then-untilize-pack (`transpose_tile` + `pack_untilize_dest`) has no helper form | n/a | 11,820 (p44, whole op) | `g6_permute_nhwc_rm_compute.cpp:66` |
| `noc_async_read` + address generator (dataflow API) | capability | No stateful read issue that takes a per-transaction NoC coordinate: `*_one_packet_set_state/with_state` fix the coordinate. A stream of same-length reads to rotating banks therefore re-programs length and control and recomputes bank, offset and XY per page. | 10,686 (`InterleavedAddrGen` reads, cursor writes) | 8,011 | `g6_permute_nhwc_rm_reader.cpp:50` |
| `noc_async_write` + address generator (dataflow API) | capability | Same gap for writes: no with_state write with a per-transaction coordinate, so each partial row pays the full issue | 8,402 (cursor reads, `InterleavedAddrGen` writes) | 8,011 | `g6_permute_nhwc_rm_writer.cpp:90` |

Each site carries its justification in the kernel-head comment.

### Artifacts
- `perf_experiments/{ablation,tensix_transpose,riscv_transpose,rm_issue_cost}/` and `tests/ttnn/unit_tests/operations/g6_permute/perf1/`. The ablation test is skipped: it ablates the pre-Perf-1 kernels.
- Bench scripts in `generated/meta/g6_permute/bench/`: `run_perf1.sh`, `run_shapes.sh`, `arm_medians.py`.
