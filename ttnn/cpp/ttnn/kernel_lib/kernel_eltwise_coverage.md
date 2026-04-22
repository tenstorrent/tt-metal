# Kernel Eltwise (FPU + SFPU) Helper Coverage

Tracks all compute kernels under `ttnn/cpp/ttnn/operations/` that perform
eltwise FPU operations (add_tiles, mul_tiles, sub_tiles, bcast, binary_dest_reuse)
and/or SFPU operations (sfpu_chain, sfpu_pipeline, raw SFPU tile calls).
Excludes tilize/untilize-only and pure copy-only kernels.

**Generated**: 2026-04-22
**Total**: 125 kernels — MIGRATED: 12 | PARTIAL: 47 | RAW: 66

Format: `STATUS | path (rel. to ttnn/cpp/ttnn/operations/) | helpers_used | raw_patterns_remaining`

Helpers: `binary_op` = binary_op_helpers.hpp, `sfpu` = sfpu_helpers.hpp/sfpu_chain,
`reduce` = reduce_helpers_compute.hpp, `dest` = dest_helpers.hpp

Raw patterns: `add_tiles`, `mul_tiles`, `sub_tiles`, `bcast_tiles` = *_tiles_bcast,
`binary_dest_reuse` = binary_dest_reuse_tiles, `sfpu_raw` = raw SFPU tile calls (exp_tile etc.)

---

## MIGRATED (12)

MIGRATED | eltwise/unary/device/kernels/compute/hardswish_kernel.cpp | binary_op,sfpu | none
MIGRATED | eltwise/unary/device/kernels/compute/hardswish_kernel_sfpu.cpp | sfpu | none
MIGRATED | eltwise/unary/device/kernels/compute/mish_kernel.cpp | binary_op,sfpu | none
MIGRATED | eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp | binary_op,sfpu | none
MIGRATED | eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp | sfpu | none
MIGRATED | eltwise/unary_ng/device/kernels/compute/hardswish_kernel.cpp | binary_op,sfpu | none
MIGRATED | eltwise/unary_ng/device/kernels/compute/mish_kernel.cpp | binary_op,sfpu | none
MIGRATED | eltwise/unary_ng/device/kernels/compute/tanhshrink_kernel.cpp | binary_op,sfpu | none
MIGRATED | normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp | binary_op,sfpu | none
MIGRATED | normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp | binary_op,reduce,sfpu | none
MIGRATED | reduction/generic/device/kernels/compute/reduce_hw_neg.cpp | sfpu | none
MIGRATED | reduction/generic/device/kernels/compute/reduce_w_neg.cpp | sfpu | none

---

## PARTIAL (47)

PARTIAL | conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp | tilize | add_tiles,mul_tiles
PARTIAL | conv/conv2d/device/kernels/conv_bmm_tilize.cpp | tilize,untilize | add_tiles,bcast_tiles
PARTIAL | experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp | reduce | add_tiles,mul_tiles,bcast_tiles,sfpu_raw
PARTIAL | experimental/conv3d/device/kernels/compute.cpp | tilize,untilize | add_tiles,bcast_tiles
PARTIAL | experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp | tilize,untilize | mul_tiles
PARTIAL | experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp | reduce | add_tiles,mul_tiles,bcast_tiles,sfpu_raw
PARTIAL | experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp | binary_op | add_tiles,mul_tiles,bcast_tiles,sfpu_raw
PARTIAL | experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp | reduce | add_tiles,mul_tiles,bcast_tiles,sfpu_raw
PARTIAL | experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp | reduce | mul_tiles
PARTIAL | experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding.cpp | tilize,untilize | add_tiles,mul_tiles,bcast_tiles
PARTIAL | experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp | binary_op | mul_tiles
PARTIAL | experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp | binary_op | mul_tiles
PARTIAL | experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp | binary_op | mul_tiles
PARTIAL | moreh/moreh_dot/device/kernels/moreh_dot.cpp | reduce | mul_tiles
PARTIAL | moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles
PARTIAL | moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles
PARTIAL | moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles
PARTIAL | moreh/moreh_norm/device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp | reduce | add_tiles,sfpu_raw
PARTIAL | moreh/moreh_norm/device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp | reduce | add_tiles,sfpu_raw
PARTIAL | moreh/moreh_norm/device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp | reduce | add_tiles,sfpu_raw
PARTIAL | moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp | reduce | add_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax/device/kernels/moreh_softmax_h.cpp | reduce | mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp | reduce | mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/groupnorm/device/kernels/compute/groupnorm.cpp | reduce,tilize,untilize | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp | reduce,tilize,untilize | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp | tilize,untilize | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp | tilize,untilize | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,binary_dest_reuse,sfpu_raw
PARTIAL | normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | normalization/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp | reduce | add_tiles,mul_tiles
PARTIAL | normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp | binary_op,sfpu | add_tiles,mul_tiles,sub_tiles,bcast_tiles
PARTIAL | normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp | reduce | mul_tiles
PARTIAL | normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp | reduce | add_tiles,mul_tiles
PARTIAL | normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp | reduce | mul_tiles
PARTIAL | normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp | reduce | add_tiles,mul_tiles
PARTIAL | reduction/generic/device/kernels/compute/reduce_h_neg.cpp | dest | sfpu_raw
PARTIAL | reduction/moe/device/kernels/compute/moe.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
PARTIAL | reduction/sampling/device/kernels/compute/sampling.cpp | reduce | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw

---

## RAW (66)

RAW | eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu.cpp | none | mul_tiles,binary_dest_reuse
RAW | eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_bcast.cpp | none | mul_tiles,binary_dest_reuse
RAW | eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp | none | binary_dest_reuse,sfpu_raw
RAW | eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp | none | sfpu_raw
RAW | eltwise/unary/device/kernels/compute/lgamma_fast_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary/device/kernels/compute/lgamma_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary/device/kernels/compute/logsigmoid_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary_ng/device/kernels/compute/hardshrink_kernel.cpp | none | binary_dest_reuse,sfpu_raw
RAW | eltwise/unary_ng/device/kernels/compute/lgamma_fast_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary_ng/device/kernels/compute/lgamma_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary_ng/device/kernels/compute/logit_kernel.cpp | none | sfpu_raw
RAW | eltwise/unary_ng/device/kernels/compute/logsigmoid_kernel.cpp | none | sfpu_raw
RAW | experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp | none | add_tiles
RAW | experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/deepseek_moe_reduce_scatter_reduction.cpp | none | add_tiles
RAW | experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp | none | add_tiles
RAW | experimental/ccl/llama_reduce_scatter_create_heads/device/kernels/compute/reduction.cpp | none | add_tiles
RAW | experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_line_reduction.cpp | none | add_tiles
RAW | experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduction.cpp | none | add_tiles
RAW | experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduction.cpp | none | add_tiles
RAW | experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp | none | add_tiles
RAW | experimental/ccl/strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | experimental/deepseek/mla/matmul_wo/device/kernels/compute_collector.cpp | none | binary_dest_reuse
RAW | experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp | none | binary_dest_reuse,sfpu_raw
RAW | experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp | none | mul_tiles
RAW | experimental/minimal_matmul/device/kernels/compute.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_reduce.cpp | none | add_tiles
RAW | experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp | none | add_tiles
RAW | experimental/reduction/integral_image/device/kernels/intimg_compute.cpp | none | add_tiles,bcast_tiles
RAW | experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp | none | add_tiles,mul_tiles
RAW | experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/ssm_eltwise_mul.cpp | none | mul_tiles,bcast_tiles
RAW | experimental/topk_router_gpt/device/kernels/compute.cpp | none | mul_tiles,sub_tiles,bcast_tiles,binary_dest_reuse,sfpu_raw
RAW | experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/reduction.cpp | none | add_tiles
RAW | experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp | none | add_tiles,mul_tiles
RAW | experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp | none | sfpu_raw
RAW | matmul/device/kernels/compute/bmm.cpp | none | mul_tiles
RAW | matmul/device/kernels/compute/bmm_large_block_zm.cpp | none | mul_tiles
RAW | matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp | none | add_tiles,bcast_tiles
RAW | moreh/moreh_abs_pow/device/kernels/moreh_abs_pow_kernel.cpp | none | sfpu_raw
RAW | moreh/moreh_adam/device/kernels/moreh_adam.cpp | none | add_tiles,mul_tiles,sub_tiles,sfpu_raw
RAW | moreh/moreh_adamw/device/kernels/moreh_adamw.cpp | none | add_tiles,mul_tiles,sub_tiles,sfpu_raw
RAW | moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp | none | mul_tiles,bcast_tiles
RAW | moreh/moreh_matmul/device/kernels/moreh_matmul.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | moreh/moreh_mean/device/kernels/moreh_mean_w.cpp | none | mul_tiles
RAW | moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp | none | add_tiles,mul_tiles,bcast_tiles
RAW | moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp | none | mul_tiles,bcast_tiles,sfpu_raw
RAW | moreh/moreh_norm/device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp | none | add_tiles,sfpu_raw
RAW | moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp | none | add_tiles,sfpu_raw
RAW | moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp | none | mul_tiles,bcast_tiles,sfpu_raw
RAW | moreh/moreh_sgd/device/kernels/moreh_sgd.cpp | none | add_tiles,mul_tiles,sub_tiles
RAW | moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp | none | add_tiles,mul_tiles,sub_tiles,sfpu_raw
RAW | moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp | none | add_tiles,mul_tiles,sub_tiles,sfpu_raw
RAW | moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/moreh_sum_nc.cpp | none | add_tiles
RAW | moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp | none | mul_tiles
RAW | moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp | none | add_tiles,bcast_tiles
RAW | normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp | none | sfpu_raw
RAW | normalization/batch_norm/device/kernels/compute/running_statistics_kernel.cpp | none | add_tiles,mul_tiles,sub_tiles
RAW | normalization/layernorm/device/kernels/compute/layernorm.cpp | none | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
RAW | normalization/layernorm/device/kernels/compute/layernorm_large_tensor_welford.cpp | none | add_tiles,mul_tiles,sub_tiles,bcast_tiles,binary_dest_reuse,sfpu_raw
RAW | normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp | none | add_tiles,mul_tiles,sub_tiles,bcast_tiles
RAW | normalization/layernorm/device/kernels/compute/layernorm_welford.cpp | none | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
RAW | normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp | none | add_tiles,mul_tiles,sub_tiles,bcast_tiles,sfpu_raw
RAW | reduction/accumulation/device/kernels/compute/accumulation_compute.cpp | none | add_tiles,mul_tiles
RAW | reduction/prod/device/kernels/compute/prod_all.cpp | none | mul_tiles
RAW | reduction/prod/device/kernels/compute/prod_nc.cpp | none | mul_tiles
