# Phase 0: Discovery Log

## Operation: layer_norm_rm
## Pattern: RM → tilize → compute → untilize → RM (Hybrid mode)

### Reference Operations Selected

| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp | RM stick reading, stick-to-tile batching |
| output_stage | untilize | ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp | Tile-to-RM writing, stick extraction |
| compute_core | batch_norm | ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp | Similar normalization pattern with gamma/beta |

### Key Design Decisions
- **Mode**: FULLY AUTOMATED
- **Input**: bfloat16, row-major interleaved, tile-aligned (32×32)
- **Output**: Same shape as input, row-major interleaved
- **Compute**: Mean → centralize → variance → inv_sqrt → standardize → optional affine
- **Optional params**: gamma (1,1,1,W), beta (1,1,1,W), epsilon (default 1e-5)
