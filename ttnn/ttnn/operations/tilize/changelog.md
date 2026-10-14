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
