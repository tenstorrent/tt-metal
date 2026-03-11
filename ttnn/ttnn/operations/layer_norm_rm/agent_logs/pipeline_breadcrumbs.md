# Pipeline Breadcrumbs — layer_norm_rm

## Phase 0: Discovery
- **Start**: Parsing user requirements
- **Operation**: layer_norm_rm
- **Math**: Layer normalization (mean → centralize → variance → inv_sqrt → normalize, optional gamma/beta affine)
- **Input**: ROW_MAJOR, interleaved, bfloat16, ≥2D, tile-aligned (32×32)
- **Output**: Same shape as input, ROW_MAJOR, bfloat16
- **Parameters**: epsilon (float, default 1e-5, keyword-only), gamma (optional tensor (1,1,1,W)), beta (optional tensor (1,1,1,W))
- **Mode**: FULLY AUTOMATED
- **Compute Detection**: RM input + compute + RM output → need tilize + compute ref + untilize

### Reference Discovery
- **Planning Mode**: Hybrid (3 references with different roles)

| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp | RM input needs in-kernel tilize |
| output_stage | untilize | ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp | RM output needs in-kernel untilize |
| compute_core | batch_norm | ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp | Closest normalization pattern (mean, variance, normalize) |

## Phase 1: Analysis
- Launched 3 analyzers in parallel
- **tilize_analysis.md**: 396 lines — reader kernel pattern for RM sticks, stick-to-tile batching
- **untilize_analysis.md**: 377 lines — writer kernel pattern, untilize helper, RM output
- **batch_norm_analysis.md**: 607 lines — compute kernel structure, CB layout, normalization pattern
- All 3 analyses COMPLETE

## Phase 2: Design
- Launching ttnn-operation-architect...
