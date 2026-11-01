# Changelog: layer_norm_tile

## Phase 0 - Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime R1 `row_resident`: tile-rows split over the full compute grid, whole 768-wide row resident in L1, optional in-kernel residual add, two-pass centered variance, gamma/beta row-0 staging for TILE or ROW_MAJOR affine in L1 or DRAM. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], alignment=[tile_aligned, h_non_aligned], residual=[residual, no_residual], affine_dtype=[bfloat16], affine_layout=[ROW_MAJOR, TILE], memory_layout=[INTERLEAVED], buffer_type=[DRAM, L1], affine_buffer_type=[DRAM, L1]. SUPPORTED == TARGET.
- **Accuracy achieved**: fp32 DEST: PCC=0.999998, max_abs_err=0.052, rel_rms_err=0.0019. bf16 DEST: PCC=0.999991, max_abs_err=0.19, rel_rms_err=0.0053 (measured on 4 shapes x residual x DEST mode via test_layer_norm_tile_precision_baseline.py).
- **Golden suite at Phase 0**: 545 / 545 passing; verifier CLI 524 supported_pass, 21 untagged regression tests passing, 0 loud categories (`verifier_results/verifier_report.json`).
- **Performance at Phase 0** (device us vs stock): 8x512 in-L1/res-DRAM/out-DRAM 55.0-55.8 vs 55.0; all-DRAM 66.6 vs 70.0; DRAM no-res 40.1-41.3 vs 48.3; all-L1 47.5-49.1 vs 43.3 (loses); L1 no-res 32.5-33.2 vs 34.5; 8x384 all-DRAM 48.0-48.4 vs 47.0 (loses); 32x32 22.3 vs 24.8; affine TILE DRAM 89.2-90.2 vs 125.4.
- **Issues encountered**: no drift. Fixed the ledger symbol-table name for the output-depth constant (`OUT_DEPTH`).
- **Tests added**: test_layer_norm_tile.py (planner), test_layer_norm_tile_precision_baseline.py, probes/probe_tail.py.

## Refinement 1 - DRAM stream efficiency on the flagged headline rows
- Date: 2026-11-01
- What was done: reader/writer rebuilt on a shared stream header (`kernels/layer_norm_tile_stream.hpp`): whole-block reserve, per-block trid ring over C-tile chunks, optional in-chunk bank rotation, grouped writer flush, and split_reader (residual read by the writer RISC on NoC1, `cb_residual_tiles` producer moves to the writer). Affine reads now issue untagged behind the block's last chunk, so the RM affine fix-up no longer delays chunk 0's push. Knobs are host constants in `layer_norm_tile_program_descriptor.py` carried on `Blocking`; split_reader is path-gated (same-tier input/residual, >= half the grid active). Measured (device kernel us, before -> after): flagged #1 54.0 -> 48.6, #2 66.2 -> 52.2, #3 39.8 -> 40.4 (flat), 8x384 all DRAM 48.8 -> 44.5, all L1 47.5 -> 31.4, L1 no-residual 32.3 -> 29.1, 32x32 22.6 -> 23.1, affine TILE DRAM 90.8 -> 77.5. Trid depth > 1, writer grouping, bank stagger and chunk 4/12/24 were flat or worse and stay parked at trivial settings.
- Accuracy achieved: unchanged; precision baseline rel-RMS ~0.0019 (fp32 DEST), PCC >= 0.99999 on [(8,1,512,768), (3,1,500,768)]; golden `(0.999, 0.02)` gate unchanged.
- Golden test progress: 75/75 on the targeted slice (8x512, 3x500, 32x32 ROW_MAJOR affine, 8x512 TILE affine, large-magnitude/mean); full suite left to the harness (545/545 at Phase 0).
- Issues encountered: deeper trid rings slow L1-interleaved sources (all-L1 39 -> 60 us at depth 3); split_reader loses ~2 us when input is in L1 and residual in DRAM, and ~1.5 us on a 32-core grid, hence the gate. Remaining headroom is the lockstep read -> compute -> write structure (see the Outcome line in op_requirements.md).
- Tests added: `tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_stream_knobs.py` (20 cases: every knob off its shipped setting on multi-block / ragged / h_non_aligned shapes).
