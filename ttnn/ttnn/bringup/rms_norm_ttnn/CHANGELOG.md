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
4. fp32 `cb_x_squared` only where it is an accumulator; `cb_normalized` back to the intermediate format (refines
   3). Changes the `fp32_dest_acc_en=True` program only; the `fp32_dest_acc_en=False` program is unchanged (same
   CBs and blocking, checked on 36 shape/operand builds on both builders).
   - What: the rule is that fp32_dest_acc_en keeps DEST 32-bit while it accumulates, and a value packed to L1 stays
     in the intermediate format unless that L1 value is itself an accumulator. (a) `cb_x_squared` is FLOAT32 only
     when the DEST fold is on (`x_squared_wt < wt_chunk`: each tile is a sum of several squares). Without the fold
     (a partial last tile, or a chunk with no divisor 2..SQ_FOLD_GROUP, e.g. 43) it holds single x^2 tiles in
     `intermediate_dtype` (`sq_dtype_of` / `_sq_dtype`). (b) `cb_normalized` is `intermediate_dtype`. (c) The
     partial-width mask follows `cb_x_squared`; a partial width never folds, so it is `intermediate_dtype`. (d)
     `CB_SQ_EXACT = true`, applied at fp32 DEST in all three solves (RESIDENT, ROW_RESIDENT, STREAM): each
     candidate chunk is priced at `cb_x_squared`'s real width and at the format that chunk would get, and
     `cb_normalized` at the intermediate tile size. At 16-bit DEST the price stays conservative (the exact price
     measured 0.987x there, see test_rms_norm_ttnn_dataflow_knobs.py). The CopySeedSfpuAdd carry from 3 is kept.
     The program-cache hash needs nothing new: the format is a function of the hashed shapes, dtypes and config.
   - Why: owner review of 3. Numbers at MiMo's (5120, 4096) bf16 weight shape, fp32 DEST, HiFi4:
     - Blocking: norm 32x4 -> 64x2 (x^2 folded in groups of 16, fp32); fused residual 32x4 -> 43x3 (prime chunk, no
       fold, bf16 x^2). L1 per core: norm `cb_x_squared` 8 -> 16 kB, `cb_normalized` 128 kB fp32 -> 128 kB bf16
       (64 tiles), total CBs 1042 -> 1178 kB; residual `cb_x_squared` 8 -> 86 kB, `cb_normalized` 128 -> 86 kB,
       total 1042 -> 1214 kB. The solve and the allocation agree to the byte.
     - Perf (1x4 mesh, 20 calls, device max): norm 0.219 -> 0.211 ms, fused residual 0.417 -> 0.413 ms.
     - MiMo golden layer 1, chunk 1 input vs float64, row-norm ratio / rel L2: with the attn_norm weight 0.99968 /
       0.00197 -> 1.00006 / 0.00232 (native 0.99945 / 0.00205, before 3 1.00090 / 0.00248); unit weight 1.00017 /
       0.00168 -> 1.00007 / 0.00167; no weight 1.00007 / 0.00167 -> 0.99999 / 0.00167. The scale bias is smaller;
       the rel L2 with a weight is back up, because y is rounded to bf16 in `cb_normalized` and again on the output.
       Random inputs with a random weight (test_precision_against_float64) show the same: rel L2 0.0018 -> 0.0024
       without a residual, 0.0024 -> 0.0029 with one; no weight unchanged (0.0017).
     - Models: MiMo 1x4 and 2x2 ladder rung last unchanged (1x4 L00-L05 0.998577 / 0.998451 / 0.998422 / 0.998510
       / 0.998787 / 0.998421, state_min 0.999281); 50k->55k chunk device 204.9 -> 204.9 ms (1x4), 249.9 -> 250.0 ms
       (2x2). Model cases: norm pcc 0.9999985 -> 0.9999972, residual-sum max rel 0.0091 -> 0.0109; both pass.
       Norm component tests and swap tests: all pass on 2x2; on 1x4 all pass except sliding_moe swaps 03-07, which
       fail on the sliding attention's per-token norm ratio upper bound 1.05 before this change too (3: [0.9742,
       1.0657], now [0.9553, 1.0691]; swap 02, whose limit is 1.08, passes).
   - Needed by: owner review of change 3.
   - Files: `device/rms_norm_ttnn_program_factory.cpp`, `rms_norm_ttnn_program_descriptor.py`,
     `tests/unit/test_rms_norm_ttnn_fp32_stats.py` (the format test for the new rule, a float64 precision test,
     scale-bias cases re-picked where the chunking moved), `tests/unit/test_rms_norm_ttnn_dataflow_knobs.py`.
