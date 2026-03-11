# Phase 0: Discovery - RMSNorm

## Operation Requirements
- **Name**: rms_norm
- **Math**: RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma
- **Inputs**: input_tensor (bfloat16/float32), gamma (optional, RM layout, shape (1,1,1,W))
- **Parameters**: epsilon (float, default 1e-6)
- **Layouts**: Both ROW_MAJOR and TILE natively (in-kernel tilize/untilize)
- **Output**: Same layout as input

## Compute Detection
- Has compute: YES (square, mean reduction, rsqrt, multiply)
- Input layout: Both RM and TILE → need tilize reference for RM path
- Output layout: Must match input → need untilize reference for RM path

## References Selected

| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp` | RM→tile conversion pattern |
| output_stage | untilize | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp` | tile→RM conversion pattern |
| compute_core | reduce_w | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp` | Reduction along W dimension (similar to RMSNorm mean along last dim) |

## Mode: Hybrid
- tilize (input) + reduce_w (compute) + untilize (output)

## Assumptions Made (Automated Mode)
1. Using multi-core interleaved tilize (not sharded) since spec doesn't mention sharding
2. Using reduce_w as compute reference since RMSNorm reduces along last dim (W)
3. Single-core not considered since multi-core is the standard path
