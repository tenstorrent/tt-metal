# rms_norm_ttnn (ttnn.bringup.rms_norm)

1. ai-generated perf optimized version
2. adding a residual add where it was needed: `return_residual_sum=True` (with `residual_input_tensor`) also returns
   the residual sum `t = x + residual`, so the separate add can go (`residual_sum_memory_config` places it; default off)
3. bug fix, fp32 statistics at `fp32_dest_acc_en=True`. This changes the default output, so it is not behind an
   option: it is the one exception to "behind an option, default off". The `fp32_dest_acc_en=False` program is
   unchanged: same kernels, same CBs, same blocking.
   - What: (a) the cross-chunk sum of squares is carried in fp32. The compute kernel's `accumulate_reduce_block`
     passes `AccumulateReloadMode::CopySeedSfpuAdd` (a new kernel constant `CARRY_RELOAD`, `CopySeedPairs` at
     16-bit DEST). (b) `cb_x_squared` and `cb_normalized` are held in fp32 (`stat_dtype` / `_stat_dtype`), as
     native `ttnn.rms_norm` holds its intermediates, and the partial-width 0/1 mask in `cb_scaler` follows
     `cb_x_squared`'s format. `cb_x_sum`, the returned residual sum, keeps its format. The L1 solve prices the two
     CBs at their new tile size, so the blocking changes at fp32 DEST. MiMo's (5120, 4096) weight shape goes from
     3 chunks of 43 to 4 chunks of 32, with or without the residual. Every build was re-solved; none failed to
     fit.
   - Why: the default reload, `CopySeedPairs`, adds an odd chunk's leftover tile with a DEST_TO_SRCB reuse add.
     That moves the fp32 carry into srcB, which is programmed to the bf16 input format, so every carry was
     truncated to bf16. Separately, x^2 was rounded to bf16 on the pack, and y was rounded twice. Measured on
     MiMo's golden layer 1, chunk 1 input with its attn_norm weight, as bringup vs float64, row-norm ratio / rel L2:
     1.00090 / 0.00248 before, 0.99968 / 0.00197 after; native was 0.99945 / 0.00205. With a unit weight the
     scale bias was +9.3e-4 and is now +1.7e-4; with no weight it was -1.6e-4 and is now +0.7e-4. The residue is
     the carry's copy_tile reload through srcA, which keeps only tf32 precision. It grows by about 0.65e-4 per
     chunk. Removing it needs an UnpackToDestFp32 accumulator, and cb_row_stat cannot be one because it is also
     an FPU operand.
   - Model cases: the MiMo norm case goes from pcc 0.9999972 / max abs 0.063 to 0.9999985 / 0.041. The
     residual-sum case goes from max rel 0.0112 to 0.0091. Both pass at their recorded limits.
   - Perf, at MiMo's shape: norm 0.215 -> 0.219 ms, fused residual 0.414 -> 0.417 ms.
   - Needed by: mimo_v2_6_d_p. The scale bias alone pushed test_swap_sliding_moe_02_attention over its ratio limit.
   - Files: `kernels/rms_norm_ttnn_compute.cpp`, `device/rms_norm_ttnn_program_factory.cpp`,
     `rms_norm_ttnn_program_descriptor.py`, and the new test `tests/unit/test_rms_norm_ttnn_fp32_stats.py`. That
     test covers both builders' CB formats and the scale bias across 2-7 chunks on device.
