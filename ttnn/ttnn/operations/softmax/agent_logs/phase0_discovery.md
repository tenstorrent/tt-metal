# Phase 0: Discovery Log

## Operation: softmax
- **Math**: exp(x_i) / sum(exp(x_j)) along dim
- **Input**: bfloat16, TILE_LAYOUT, 4D (N, C, H, W)
- **Output**: same shape/dtype/layout
- **Parameters**: dim (-1 or -2), numeric_stable (bool, default True)

## Compute Detection
- Input layout: Tile
- Has compute: Yes (exp, reduce_sum, reciprocal, multiply)
- Output layout: Tile
- References needed: compute_core only (no tilize/untilize)

## References Selected

| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| compute_core | tt-train softmax | tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp | Direct softmax impl |
| compute_core (W reduction) | reduce_op_multi_core_w | ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp | dim=-1 pattern |
| compute_core (H reduction) | reduce_op_multi_core_h | ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp | dim=-2 pattern |

## Mode
- Planning mode: Hybrid (combining softmax compute with W/H reduction patterns)
- Automation: FULLY AUTOMATED
- Breadcrumbs: ENABLED
