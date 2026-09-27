# rms_norm_ttnn (ttnn.bringup.rms_norm)

1. ai-generated perf optimized version
2. adding a residual add where it was needed: `return_residual_sum=True` (with `residual_input_tensor`) also returns
   the residual sum `t = x + residual`, so the separate add can go (`residual_sum_memory_config` places it; default off)
