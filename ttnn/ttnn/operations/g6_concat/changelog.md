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
