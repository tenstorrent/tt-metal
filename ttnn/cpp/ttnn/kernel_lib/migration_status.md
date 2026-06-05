# eltwise convenience migration — status ledger

Scoreboard for the per-kernel before/after review at `ttnn/cpp/ttnn/kernel_lib/docs/migration_log.html` (serve over HTTP; see the file header).

- **Before base:** `d0f737b9497` (pre-eltwise) · **After:** working tree

- **Total kernels:** 100 — **92 full**, **8 partial**


## By group

| group | kernels | full | partial |
|---|--:|--:|--:|
| eltwise | 15 | 15 | 0 |
| copy | 1 | 1 | 0 |
| data_movement | 5 | 5 | 0 |
| normalization | 19 | 16 | 3 |
| moreh | 31 | 30 | 1 |
| reduction | 2 | 2 | 0 |
| experimental | 15 | 11 | 4 |
| rand/uniform | 2 | 2 | 0 |
| kernel_lib tests | 6 | 6 | 0 |
| toy/examples | 3 | 3 | 0 |
| misc | 1 | 1 | 0 |

## Partial migrations (some stages remain raw LLK)

- `normalization/groupnorm/device/kernels/compute/groupnorm.cpp` — **(revisit 0106)** the sub_tiles_bcast_scalar residual loops, the (Var+eps)→rsqrt block, the input-mask mul (Block×Row), both re-mask muls, the (x−E[x])² square, and the (x−Ex)·1/√ scalar-bcast mul are now all on the chain (subblock slack mirrors the existing migrated sub-stages). Only the **copy-or-add accumulate** and the **optional gamma/beta** loops remain raw — all three select copy-vs-FPU-op per tile from a runtime variable (`copy_or_add` / `apply_gamma_beta[]`), which the compile-time-op chain cannot express. The reduce helpers stay (separate family).
- `normalization/layernorm/device/kernels/compute/layernorm.cpp` — Three FPU stages migrated to the eltwise convenience layer: the fused pre-add X+Y, the x-E[x] col-bcast sub, the (x-E[x])^2 squaring, and the Var+eps &rarr; rsqrt block; the gamma/beta scale loop stays raw LLK.
- `normalization/layernorm/device/kernels/compute/layernorm_large_tensor_welford.cpp` — Two stages of the Welford layernorm — add+rsqrt+pack and the COL unary_bcast — migrated to eltwise_chain/unary_bcast; the Welford LLK and second-pass bcast loops stay raw.
- `moreh/moreh_mean/device/kernels/moreh_mean_w.cpp` — Only the do_mask_w masking block is migrated to a CopyTile &rarr; CopyTile &rarr; Mask &rarr; PackTile chain; the matmul-based W reduction stays raw LLK.
- `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp` — Only the 1/sqrt(mean_squared + eps) block is migrated to a fixed BinaryFpu(Add) &rarr; Rsqrt &rarr; PackTile chain; the norm-x, weight, and ROPE-fusion loops stay raw LLK.
- `experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp` — The three FPU eltwise loops (rotated*sin, x*cos, cos_interim+sin_interim) migrate to mul&lt;&gt;/mul&lt;&gt;/add&lt;&gt; (each BinaryDataFormatReconfig::Input + PackTileReconfig::None); the x @ trans_mat matmul loop stays raw LLK.
- `experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp` — Only the sin_interim = rotated * sin ROW-bcast loop migrates to compute_kernel_lib::mul&lt;&gt;. The matmul, cos, and add stages stay raw LLK because they read runtime-selected CBs (in_cb/out_cb), which the chain templates can't accept under the TRISC2 code-size budget.
- `experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` — Single-tile RM variant: only the sin_interim = rotated * sin stage migrates to compute_kernel_lib::mul&lt;&gt;. The matmul, cos, and add stages stay raw LLK due to runtime-selected CBs and the TRISC2 size budget.

## All kernels

| kernel | badge | Δ | decisions |
|---|---|---|--:|
| `eltwise/binary_ng/device/kernels/compute/eltwise_where_no_bcast.cpp` | full | +80/-59 | 5 |
| `eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu.cpp` | full | +60/-50 | 5 |
| `eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.cpp` | full | +51/-40 | 4 |
| `eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp` | full | +39/-46 | 4 |
| `eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp` | full | +56/-40 | 5 |
| `eltwise/unary/device/kernels/compute/eltwise_identity_kernel.cpp` | full | +12/-25 | 4 |
| `eltwise/unary/device/kernels/compute/hardswish_kernel.cpp` | full | +78/-42 | 5 |
| `eltwise/unary/device/kernels/compute/lgamma_fast_kernel.cpp` | full | +66/-90 | 5 |
| `eltwise/unary/device/kernels/compute/lgamma_kernel.cpp` | full | +86/-120 | 6 |
| `eltwise/unary/device/kernels/compute/logit_kernel.cpp` | full | +60/-58 | 5 |
| `eltwise/unary/device/kernels/compute/logsigmoid_kernel.cpp` | full | +33/-37 | 5 |
| `eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` | full | +60/-46 | 5 |
| `eltwise/unary/device/kernels/compute/where_tss_kernel.cpp` | full | +71/-42 | 5 |
| `eltwise/unary_backward/gelu_bw/device/kernels/compute/eltwise_bw_gelu_poly.cpp` | full | +46/-41 | 4 |
| `eltwise/unary_backward/tanh_bw/device/kernels/compute/eltwise_bw_tanh_deriv.cpp` | full | +42/-40 | 4 |
| `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` | full | +16/-27 | 3 |
| `data_movement/bcast/device/kernels/compute/bcast_h.cpp` | full | +34/-26 | 4 |
| `data_movement/bcast/device/kernels/compute/bcast_hw.cpp` | full | +40/-28 | 3 |
| `data_movement/bcast/device/kernels/compute/bcast_w.cpp` | full | +35/-21 | 3 |
| `data_movement/clone/device/kernels/compute_kernel.cpp` | full | +18/-19 | 4 |
| `data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` | full | +16/-21 | 4 |
| `normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp` | full | +159/-162 | 6 |
| `normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp` | full | +109/-217 | 6 |
| `normalization/batch_norm/device/kernels/compute/running_statistics_kernel.cpp` | full | +99/-29 | 5 |
| `normalization/batch_norm/device/kernels/compute/running_statistics_sfpu_kernel.cpp` | full | +120/-259 | 6 |
| `normalization/groupnorm/device/kernels/compute/groupnorm.cpp` | partial | +90/-63 | 5 |
| `normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp` | full | +205/-161 | 6 |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp` | full | +88/-69 | 5 |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp` | full | +123/-90 | 6 |
| `normalization/layernorm/device/kernels/compute/layernorm.cpp` | partial | +142/-69 | 6 |
| `normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp` | full | +45/-32 | 5 |
| `normalization/layernorm/device/kernels/compute/layernorm_large_tensor_welford.cpp` | partial | +42/-32 | 5 |
| `normalization/layernorm/device/kernels/compute/layernorm_welford.cpp` | full | +152/-140 | 6 |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp` | full | +79/-46 | 5 |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp` | full | +31/-15 | 3 |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp` | full | +21/-30 | 5 |
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | full | +83/-83 | 6 |
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | full | +20/-30 | 4 |
| `normalization/softmax/device/kernels/attention/compute/softmax.cpp` | full | +117/-91 | 6 |
| `normalization/softmax/device/kernels/attention/compute/softmax_sharded.cpp` | full | +50/-43 | 5 |
| `moreh/moreh_adam/device/kernels/moreh_adam.cpp` | full | +198/-130 | 6 |
| `moreh/moreh_adamw/device/kernels/moreh_adamw.cpp` | full | +157/-91 | 6 |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/moreh_clip_grad_norm_step1_kernel.cpp` | full | +217/-93 | 6 |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/moreh_clip_grad_norm_step2_kernel.cpp` | full | +116/-44 | 6 |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/moreh_clip_grad_norm_step3_kernel.cpp` | full | +29/-34 | 5 |
| `moreh/moreh_dot/device/kernels/moreh_dot.cpp` | full | +16/-23 | 4 |
| `moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp` | full | +30/-27 | 6 |
| `moreh/moreh_mean/device/kernels/moreh_mean_h.cpp` | full | +27/-19 | 3 |
| `moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp` | full | +31/-34 | 4 |
| `moreh/moreh_mean/device/kernels/moreh_mean_w.cpp` | partial | +28/-20 | 4 |
| `moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp` | full | +59/-40 | 5 |
| `moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/moreh_nll_loss_step2_kernel.cpp` | full | +103/-135 | 6 |
| `moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp` | full | +88/-92 | 6 |
| `moreh/moreh_norm/device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` | full | +227/-102 | 6 |
| `moreh/moreh_norm/device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp` | full | +184/-90 | 6 |
| `moreh/moreh_norm/device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` | full | +212/-101 | 6 |
| `moreh/moreh_norm/device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` | full | +113/-125 | 5 |
| `moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp` | full | +65/-105 | 6 |
| `moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` | full | +107/-123 | 6 |
| `moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp` | full | +235/-127 | 6 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp` | full | +207/-71 | 6 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_h.cpp` | full | +118/-81 | 5 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp` | full | +217/-52 | 5 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp` | full | +106/-82 | 5 |
| `moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp` | full | +211/-54 | 6 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp` | full | +80/-25 | 5 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp` | full | +152/-32 | 5 |
| `moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp` | full | +143/-29 | 6 |
| `moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp` | full | +34/-23 | 5 |
| `moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp` | full | +26/-22 | 5 |
| `moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp` | full | +47/-22 | 5 |
| `reduction/prod/device/kernels/compute/prod_all.cpp` | full | +48/-61 | 6 |
| `reduction/prod/device/kernels/compute/prod_nc.cpp` | full | +40/-37 | 5 |
| `experimental/bcast_to/device/kernels/compute/compute_interleaved_col_bcast_to.cpp` | full | +14/-13 | 4 |
| `experimental/bcast_to/device/kernels/compute/compute_interleaved_row_bcast_to.cpp` | full | +10/-13 | 2 |
| `experimental/bcast_to/device/kernels/compute/compute_interleaved_scalar_bcast_to.cpp` | full | +10/-13 | 4 |
| `experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp` | full | +93/-95 | 5 |
| `experimental/dropout/device/kernels/compute/dropout_kernel.cpp` | full | +29/-32 | 3 |
| `experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp` | full | +80/-49 | 6 |
| `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp` | partial | +36/-16 | 4 |
| `experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding.cpp` | full | +87/-66 | 6 |
| `experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp` | full | +37/-61 | 6 |
| `experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp` | full | +104/-38 | 6 |
| `experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp` | partial | +43/-32 | 5 |
| `experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp` | partial | +26/-8 | 4 |
| `experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` | partial | +21/-6 | 4 |
| `experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` | full | +83/-94 | 4 |
| `experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_poly.cpp` | full | +43/-36 | 5 |
| `rand/device/kernels/compute_uniform.cpp` | full | +12/-13 | 4 |
| `uniform/device/kernels/compute_uniform.cpp` | full | +13/-13 | 4 |
| `kernel_lib/tests/chain_reconfig/chain_2arg_combined.cpp` | full | new | 4 |
| `kernel_lib/tests/chain_reconfig/chain_4arg_with_dt.cpp` | full | new | 4 |
| `kernel_lib/tests/chain_reconfig/chain_elide.cpp` | full | new | 3 |
| `kernel_lib/tests/chain_reconfig/chain_mixed_prev.cpp` | full | new | 3 |
| `kernel_lib/tests/chain_reconfig/chain_pack_to_bfp8.cpp` | full | new | 4 |
| `kernel_lib/tests/chain_reconfig/chain_singleside.cpp` | full | new | 4 |
| `toy_binary_in_place/kernels/compute.cpp` | full | new | 5 |
| `toy_variance/kernels/compute.cpp` | full | new | 6 |
| `ttnn/examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp` | full | +10/-66 | 3 |
| `kernel/compute/eltwise_copy.cpp` | full | +17/-20 | 4 |
