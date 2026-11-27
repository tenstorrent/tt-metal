# Changelog: g6_concat

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Two regimes, both multi-core over output-page ranges on the full grid (110 cores on this Blackhole):
  - `page_copy`: one input page per output page, rows `[0, v)` only for short last tile rows;
  - `row_gather`: h joins with unaligned offsets, assembled from row-range reads plus a core-local hop for odd shifts.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT], concat_dim=[outer, h, w], alignment=[tile_aligned, h_non_aligned], input_buffer=[dram], output_buffer=[dram]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio 1.0 at p5/p50/p95, measured on 4 shapes via `test_g6_concat_precision_baseline.py` (one tile row, 31-row images, `row_gather` Qwen 79, multi-block Qwen 23)
- **Golden suite at Phase 0**: 79 / 80 supported cells passing, 86 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode (per `pipeline_logs/phases/verifier/verifier_report.json`). The one supported_fail and 2 of 3 regression failures come from golden-harness defects (bf16 NaN-canonicalizing oracle; `eval/profiling.py` under trace), documented in `verification_report.md`.
- **Performance at Phase 0**: p2/p3 at 1.025×/1.027× of the 460.8 GB/s DRAM roofline, p4 at 1.035×. p1 (fp32, measured with `validate()` bypassed) at 1889 µs, against 1950 µs for stock and the 1747.6 µs roofline.
- **Issues encountered**: verifier fixes:
  - the program-cache key now includes every blocking knob (`BLOCK_BYTES_CAP`, `OUTPUT_PAGES_DEPTH`, `TRIM_ROWS_MAX`, `MAX_CORES`) (BB-07);
  - stale host-symbol names in kernel comments corrected;
  - two broken and two duplicate perf harnesses, plus an empty `tools/zones.py`, removed in favour of one `test_g6_concat_device_perf.py`.
- **Tests added**: `test_g6_concat.py` (acceptance, 47 cases), `test_g6_concat_precision_baseline.py`, `test_g6_concat_device_perf.py` (profiler harness; p1 skips until float32 is supported)

## Refinement 1 — dtypes float32 + bfloat8_b, interleaved L1 / mixed placement
- **Date**: 2026-11-27
- **What was done**:
  - `SUPPORTED` gains `dtype` float32 and bfloat8_b, `input_buffer` l1 and mixed, and `output_buffer` l1.
  - `EXCLUSIONS` gains `{bfloat8_b, h, h_non_aligned}`, the deferred `row_gather_bfp8` regime: the NoC cannot place the 1-byte face-row exponents. No INPUTS cell falls in it.
  - Per-input placement (design: *Interleaved accessor buffer type*). Both readers take `[l1_mask_lo, l1_mask_hi, dram accessor CT, l1 accessor CT]`; bit k of the mask is set when input k is in L1. When a buffer type has no inputs, its slot repeats the other type's args. The kernel reads that slot's `is_dram`, so all-DRAM and all-L1 calls compile to one accessor and the mask test folds away. The shared header is `kernels/g6_concat_input_placement.hpp`.
  - `page_copy` reader: the fetch loop now walks *runs* of pages from one input. The accessor is picked once per run (`with_accessor`), with no per-page branch on placement. The per-page body moved into `fetch_run` unchanged.
  - `row_gather` reader: the read alignment is now per input (L1 16 B, DRAM 64 B), so L1 pieces are always direct. The `has_staging` static_assert holds whenever there are no DRAM inputs. Staging is still allocated only for DRAM inputs with `rb mod 64 ≠ 0` (bf16).
  - L1 output: `build_plan`'s DRAM-output assert is removed. `out_l1_per_bank = ceil(P_out / num_l1_banks) · page_bytes` is added to `l1_total` for the BB-05 check and reported by `g6_concat_plan`. The writer was already using the output tensor's accessor.
  - Reused: both readers, the shared writer, the `out_page` split, block and depth knobs, CB formats (already taken from the tensor dtype), and `TRIM_ROWS_MAX[bfloat8_b] = 0` (whole pages). Added: one header (placement selection, shared by both readers) and the per-input mask CT words.
  - `/numeric-formats-metal`: only its dtype-surface part applies. There is no compute kernel, and the spec forbids `compute_kernel_config`, so the op has no `fp32_dest_acc_en` axis, no intermediate CB and no `UnpackToDestFp32`. There is no precision matrix over fidelity, because the op does no arithmetic and every gate is bit-exact.
- **Accuracy achieved**: bit-exact (PCC = 1.0, relative RMS = 0) on every pin. Shapes: fp32 / L1 / mixed `row_gather` with shift 31 `[(1,2,31,64),(1,2,33,64),(1,2,7,64)]`; bf8_b w / outer joins on DRAM; Panoptic-like bf8_b mixed → L1 `5×(1,4,64,256)`; bf16 DRAM → L1; fp32 trimmed w join; 40 inputs with mixed placement (mask word 2).
- **Golden test progress**: targeted slice of 57 tests. All 49 target cells pass: 40 fp32 INPUTS, p1, the fp32 special-values case, Qwen 157/158, the L1 poisoned-padding loose case, and Panoptic 23/32/40/51. `test_cache_key_placement` runs all three steps, and the bf16 `row_gather` representatives (Qwen 79/89/143) still pass. The 1 xfail is Qwen 159 (ROW_MAJOR, Refinement 2). The full golden run is left to the harness.
- **Performance (no-regression check, device kernel ns, 110 cores)**: p1 fp32 1,890,662 (verifier baseline 1,889 µs), p2 902,270, p3 899,504, p4 451,517 (Phase 0: 455 µs), Qwen 79 125,336 (Phase 0: 124.5 µs). The run-based `page_copy` loop is neutral.
- **L1**: Panoptic 23 plan: block 24, `l1_total` 78,336 B (CB 52,224 B, output 26,112 B per bank). Panoptic 40 plan: block 84, `l1_total` 274,176 B. Both use 110 L1 banks.
- **Issues encountered**: None.
- **Tests added**: `test_g6_concat_dtypes_placement.py` (15 cases): `row_gather` × {fp32 DRAM, fp32 mixed → L1, bf16 L1, bf16 mixed × 2}; `page_copy` × {bf8_b w / outer on DRAM, bf8_b mixed → L1, bf8_b L1 → L1, fp32 trimmed w, bf16 mixed trimmed w, bf16 DRAM → L1, fp32 mixed h}; 40 mixed inputs; the plan's L1-output bytes.

## Refinement 2 — ROW_MAJOR layout (native stick copy)
- **Date**: 2026-11-27
- **What was done**:
  - `SUPPORTED["layout"]` gains `ROW_MAJOR_LAYOUT`. `EXCLUSIONS` gains `{layout: ROW_MAJOR, concat_dim: w}`, the deferred `stick_gather_row_major` regime: a last-dim join cuts inside every stick. No INPUTS cell is in it.
  - New host regime `page_copy_row_major` (`select_regime`: any ROW_MAJOR call). It runs on the existing page reader and shared writer with page = stick. The address map is `(O, C_k)` in stick units: `O = Π s[0:d]`, `C_k = n_k · Π s[d+1:r−1]`. `P_out = Π out[0:r−1]`, with the same `out_page` split over `min(P_out, grid)` cores and the same `block_pages = min(n1, BLOCK_BYTES_CAP // page_bytes)` knob, depth and CB.
  - `page_bytes` is now `buffer_aligned_page_size()` for both layouts (unchanged for tiles). The host asserts that every input and the output share it and that it is a multiple of the DRAM read alignment, so every CB slot keeps the base's 64 B alignment. A stick is `W · elem` with `W % 32 == 0`, so this holds whenever the gate passes.
  - Trimming is off for RM: `trim_rows_max = 0`, and the row geometry is degenerate (`out_wt = out_ht = 1`, 32 valid rows). `is_trimmed` now takes `trim_rows_max` as a template parameter, and `0` folds to "whole page" at compile time. This affects RM and bf8_b, and leaves the row cursor dead code there. The bf16/fp32 TILE binaries are unchanged in behavior.
  - Per-input placement: R1's L1 mask and dual accessor carry over unchanged. Qwen 159 (DRAM + L1 RM inputs) passes.
  - Reused: the page reader, shared writer, `out_page` split, block and depth knobs, placement mask, CB and plan report. Added: one regime name, the stick branch of the address map and plan geometry, and the compile-time trim fold.
- **Accuracy achieved**: bit-exact (PCC = 1.0, relative RMS = 0) on every pin, including:
  - outer joins: `(2,3,2560)+(14,3,2560)`, 16×`(1,3,2560)`, rank-4 dim 1;
  - `h` joins: `(16,2,2560)+(16,1,2560)`, rank-4 dim 2 with 3 inputs;
  - fp32: outer `(3,5,96)+(2,5,96)` and `h` `(2,7,2560)+(2,33,2560)`;
  - Qwen 159 mixed placement, mixed → L1 output and fp32 L1 → L1;
  - a multi-block ragged case: 6600 sticks, 60 per core, block 25;
  - a 196,608 B stick wider than `BLOCK_BYTES_CAP`.
  The 12 RM pins are also clean under `--dev` (watcher, NoC sanitizer).
- **Golden test progress**: targeted slice of 42 tests, all passing: all 37 RM cells (36 DRAM, plus Qwen 159 with mixed inputs), plus these prior-passing representatives on the edited code paths: Qwen 79 (`row_gather`, golden and loose), Panoptic 23 (bf8_b, mixed → L1), `test_cache_key` and `test_cache_key_placement`. The full golden run is left to the harness.
- **Performance**: not measured. Every RM production cell has ≤ 48 sticks (≤ 32 tiles), and `perf_rules.txt` exempts that size. The plan runs one stick per core on 48 cores (2·5120 B CB). The TILE page reader and writer differ only by the compile-time trim fold.
- **Issues encountered**: None.
- **Tests added**: `test_g6_concat_row_major.py` (46 cases):
  - 12 RM pins (outer, `h`, fp32, mixed / L1 placement, multi-block, wide stick);
  - the RM plan report;
  - the RM last-dim `ExcludedCell` refusal;
  - `test_g6_concat_layout_matrix`: TILE and ROW_MAJOR × bf16 / fp32 × 8 shapes (aligned, H non-aligned, `row_gather`-triggering offsets).
  W stays tile-aligned in the matrix, because `w_non_aligned` has no TARGET cell.

## Refinement 3 — Speed up the perf-focus profile (p1, Qwen 18)
- **Date**: 2026-11-27
- **What was done**:
  - **New knob `TRANSFER_BYTES_MAX = 2048`.** A whole page larger than the knob moves as transfers of at most that size, issued piece-major over the run/block in both the page reader and the shared writer: piece 0 of every page, then piece 1, and so on. Consecutive pages of an interleaved buffer sit in consecutive banks, so consecutive transfers now rotate DRAM banks, where before each bank got 4 KB at a time. `transfer_bytes = page_bytes` when a block holds a single page, since its pieces would hit one bank back to back.
    - Pages that are not split: bf16 and bf8_b tiles (≤ 2 KB), and ROW_MAJOR sticks ≤ 2 KB.
    - Pages that are split: fp32 tiles (2 pieces) and wider sticks, for example 5120 B → 2048 / 2048 / 1024.
    - Trimmed pages move their valid rows once, in the piece-0 sweep.
  - **New knob `BANK_STAGGER`** (catalog `bank_stagger`, default `False`). Rotates each core's in-run issue order by `core_idx mod num_dram_banks` (per-core runtime arg).
  - **New knob `READ_TRIDS`** (catalog `trid_double_issue`, default `0`). A page-reader ring of T blocks tagged with transaction ids 1..T, the oldest retired by its own barrier. The landing slot is `write_ptr + in_flight` blocks, wrapped by hand. Needs `2 ≤ T ≤ OUTPUT_PAGES_DEPTH`.
  - Shared header `kernels/g6_concat_page_sweep.hpp` (`sweep_run`, `walk_run`) holds the issue order for both kernels. All knobs are in `blocking_knobs()` (program-cache key), and the plan report prints `transfer_bytes`, `bank_stagger` and `read_trids`.
  - Reused: page reader run walk, shared writer, placement mask, trim path, out_page split, block/depth knobs, CB. Added: one header, 3 knobs, 4 reader and 2 writer CT words, one RT word per kernel.
- **Measurement** (p1, device kernel ns, 110 cores, bit-exact every row; `test_g6_concat_perf_sweep.py`):

  | Setting | p1 ns | vs whole pages |
  |---|---|---|
  | whole pages (pre-R3 behaviour) | 1,894,776 | — |
  | **default: 2 KB piece-major** (median of 3: 1,799,191) | 1,802,341 | **−4.9%** |
  | 1 KB transfers | 1,838,310 | −3.0% |
  | cap 64 / 256 KiB | 1,797,781 / 1,811,418 | noise |
  | depth 3 / 4 | 1,795,242 / 1,801,394 | noise |
  | bank stagger, 2 KB / whole pages | 1,799,057 / 1,893,410 | null |
  | trid 2 @ d2, 2 @ d3, 3 @ d4, 4 @ d4 | 1,818,307 / 1,804,696 / 1,802,660 / 1,813,623 | null |
  | trid 3 @ d4, cap 32 KiB (median of 3) | 1,785,207 | −0.8% vs default, below noise |
  | whole pages + trid 3 @ d4 | 1,896,541 | null |

  Before the clean kernels, I explored with throwaway ablations (whole-page transfers, same ns basis):
  - **Pre-R3 cap × depth sweep** (16 KiB … 512 KiB × d2–d4): 1,866–1,923 µs. Larger blocks were slower.
  - **Read-only / write-only:**
    - before: 976 / 1,337 µs;
    - after: 972 / 1,303 µs;
    - sync-only: 14 µs;
    - bf16 p2: 485 / 650 µs.
  - **NoC swap** (reads NOC1, writes NOC0): 2,415 µs.
  - **Core count** 64 / 80 / 96: 1,868 / 1,880 / 1,899 µs. With 1 KB reads, all three sit at ≈ 1,837 µs.
  - **Page-major split** (pieces of one page back to back): reads 1 KB 1,824 µs; writes 1 KB 1,900 µs.
  - **Bank-coalesced 16 KB writes** (pages p, p+8, … are contiguous in a bank): 2,109 µs, write-only 1,447 µs.
  - **Piece-major 1 KB on bf16 p2:** 918 vs 900 µs.
- **Guard set** (old vs new kernels, same session, µs):

  | Cell | Old | New |
  |---|---|---|
  | p1 | 1,901 / 1,888 | 1,804 / 1,807 |
  | p2 | 897 / 900 | 900 / 902 |
  | p3 | 900 / 895 | 901 / 898 |
  | p4 | 452.8 / 452.7 | 454.5 / 456.2 |
  | Qwen 79 (`row_gather`) | 125.0 / 124.6 | 124.4 / 126.2 |
  | Qwen 80 (trimmed w) | 10.4 / 10.1 | 10.2 / 10.6 |
  | Qwen 25 (median of 6) | 10.42 | 10.44 |
  | Panoptic 23 (bf8_b mixed → L1) | 8.1 / 8.1 | 8.1 / 8.0 |
  | Panoptic 40 (bf8_b L1 → L1, median of 6) | 44.3 | 44.3 |

  No RM production cell exceeds 32 tiles, so none is measured. Their blocks hold one stick, so `transfer_bytes = page_bytes` and their behaviour is unchanged.
- **Bottleneck now**: DRAM efficiency under mixed read/write traffic, at 448 GB/s (1.03× the 460.8 GB/s roofline). This is p2's per-byte rate. Writes alone cap at 309 GB/s: every NOC1 write to a channel lands on one endpoint. Remaining headroom is in the noise band.
- **Accuracy achieved**: bit-exact (PCC = 1.0, relative RMS = 0) on p1 under all 16 sweep settings, and on 12 knob pins: fp32 trimmed w join, bf16 outer, fp32 ROW_MAJOR 5120 B sticks × {stagger, trid 2, trid 3 @ d4 + stagger, 1 KB + stagger}, on 7 cores with ragged blocks.
- **Golden test progress**: targeted slice of 93/93 passing: all 42 FLOAT32 cells (including p1 golden and loose, and fp32 special values), the 4 bf8_b Panoptic cells, all 37 ROW_MAJOR cells, Qwen 79 golden/loose, `test_cache_key`, `test_cache_key_placement` and `test_repeat_call`. The full run is left to the harness. Unit directory: 112/112 (acceptance, dtypes/placement, row-major, precision).
- **Issues encountered**: the first version of the p1 sweep fixture cached ttnn tensors at module level. Precompile's collect pass runs test bodies on a remote-only capture MeshDevice, so the real pass reused tensors from that device and threw in `MeshBuffer::create`. Now only the torch inputs are cached, and device tensors are rebuilt per `device` object (unit dir 152/152 with precompile). The `--profile` wrapper also splits `-k` expressions on spaces, so profiling used the `G6C_SWEEP` id selector.
- **Tests added**:
  - `test_g6_concat_perf_sweep.py`: the p1 knob sweep (16 settings, bit-exact) and `test_g6_concat_knob_pins` (12 cases).
  - `test_g6_concat_device_perf.py` gains Panoptic 23 / 40 (bf8_b, L1 placement) for the guard set.
