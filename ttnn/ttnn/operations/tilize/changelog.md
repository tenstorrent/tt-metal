# Changelog: tilize

## Phase 0 — Core Implementation
- **Date**: 2026-10-14
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). `interleaved_row_split` regime: tile-rows split over the full runtime grid, `compute_kernel_lib::tilize`, bank-coalesced double-buffered staging reader with bank stagger, flush-batched writer.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], output_dtype=[bfloat16], in_tile_height=[none] (ROW_MAJOR input), tile_height=[32,16,8,4,2,1], alignment=[tile_aligned], rank=[2,3,4,5,6], tile_grid=[single_tile, small, tall_narrow], low_l1=[False, True], buffer=[dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram] (interleaved), shard_api/orientation/pad_mode/pad_value=[none]
- **Accuracy achieved**: PCC=1.0, max_abs_err=0, rms_err=0, max ULP 0 — bit-identical (measured on 5 shapes via test_tilize_precision_baseline.py)
- **Golden suite at Phase 0**: 238 supported_pass, 0 supported_fail, 0 xpass_drift; 56 xfail_wrong_mode are harness-side fp8 input-creation failures (per `verifier_report.json`)
- **Perf (BH, 110 cores)**: `[1,1,16384,64]` 11.35 µs (reference 11.85 µs)
- **Issues encountered (verifier fixes)**: SUPPORTED under-claimed low_l1/buffer/rank/tile_height (verified on device, added); entry point dropped `tile=` on the output allocation (fixed); validate() raised ValueError for unaligned TILE (retile) inputs instead of a support refusal (fixed); runtime shard-side tagging read legacy shards as nd (fixed); L1 ledger total missing the staging CB (fixed).
- **Tests added**: test_tilize.py (tiny tile, buffers × low_l1, ranks), test_tilize_precision_baseline.py

## Refinement 1 — Full work geometry: 2-D split + padding
- Date: 2026-10-14
- What was done:
  - `interleaved_2d_split`: host selector (`_choose_split`) picks `cores_r × cores_c ≤ grid` minimizing the busiest core's tile count when `R < grid` (row-only otherwise — Phase 0 path, coalesced reader, unchanged). Columns split on read-alignment units; cores grouped by column extent (≤ 2 groups), one reader/compute/writer + CB set per group. Knob `ENABLE_2D_SPLIT`.
  - Segment-reader bank stagger: each core starts its stick walk at stick `k % tile_h` (knob `SEGMENT_BANK_STAGGER`; ~8 % on short_wide, ~2 % on square_large).
  - Padding in `load_sticks_block`: output tile grid = padded shape; each segment reads only its in-range bytes; W tail, H-tail sticks, pad images (leading-dim padding, ≤ 4 leading dims) and whole pad tile-columns filled by local L1→L1 NoC copies from `cb_pad_fill` (pad value in the input format, bit-cast for ints, torch-exact for floats). Rank 0/1 synthesize `[tile_h, 32]` / `[tile_h, round(W,32)]`.
  - Output allocated at the padded shape, then a metadata-only view to the logical shape. Added a Python overload `ttnn.experimental.view(tensor, logical_shape, padded_shape)` (ttnn/cpp/ttnn/operations/experimental/reshape/view_nanobind.cpp, wraps the existing C++ `view`; requires rebuilding `_ttnn.so`).
  - SUPPORTED: tile_grid += short_wide, square_large; pad_mode += auto, explicit; pad_value += zero, positive, negative; alignment += w/h/hw_non_aligned; rank += 0, 1.
  - Reused: reader/compute/writer kernels and per-core rectangle RT args, compute_blocking, CB layout. Added: selector + column groups (host), pad branch in the segment reader, `cb_pad_fill`.
- Accuracy achieved: bit-exact (torch.equal) on all shapes tested, incl. pad region; PCC=1.0, rtol=0, atol=0.
- Golden test progress: targeted slice 31/31 (26 bf16 interleaved pad/short_wide/square_large cells incl. 2 loose cases, 5 program-cache cases); translated padding/to_layout slice 421 passed, 82 xfailed (non-bf16 dtypes, SUPPORTED gate).
- Perf (BH p150b, per-timer-cluster span from profile_log_device.csv): `[1,1,32,2048]` 64 cores 4.1 µs (row-only 1 core: 5.6 µs); `[1,1,32,8192]` 86 cores 6.1 µs (12.2 µs); `[1,1,2048,2048]` 110 cores 39.6 µs (64 cores: 43.2 µs); perf-focus `[1,1,16384,64]` unchanged at ~11.2–11.4 µs (coalesced path untouched).
- Issues encountered: no Python API to build a TILE tensor whose padded shape exceeds the tile round (explicit padding) — solved with the view binding overload. Profiler CSV `DEVICE KERNEL DURATION` is invalid on 110-core runs on this board (offset timer at x=11); spans computed per timer cluster.
- Tests added: `tests/ttnn/unit_tests/operations/tilize/test_tilize_geometry_padding.py` (split coverage/core count; ragged W tail on column groups, single-stick, low_l1, tiny tile + H tail, leading-dim pad, rank growth, odd row bytes); short_wide/square_large shapes added to `test_tilize_perf.py`.
