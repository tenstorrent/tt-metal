// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/adaptive_pool/adaptive_pools_nanobind.hpp"
#include "ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_nanobind.hpp"
#include "ttnn/operations/experimental/cnn/convert_to_hwc/convert_to_hwc_nanobind.hpp"
#include "ttnn/operations/experimental/conv3d/conv3d_nanobind.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_nanobind.hpp"
#include "ttnn/operations/experimental/reduction/integral_image/intimg_nanobind.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate_nanobind.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write_nanobind.hpp"
#include "ttnn/operations/experimental/ssm/hc_sum_reduce/hc_sum_reduce_nanobind.hpp"
#include "ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_nanobind.hpp"
#include "ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/nlp_create_qkv_heads_boltz_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice_nanobind.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/rmsnorm_distributed_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/dit_layernorm_pre_all_gather_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/dit_layernorm_post_all_gather_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/rotate_half/rotate_half_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.hpp"
#include "ttnn/operations/experimental/copy/typecast/typecast_nanobind.hpp"
#include "ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_nanobind.hpp"
#include "ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/ccl_experimental_nanobind.hpp"
#include "ttnn/operations/experimental/plusone/plusone_nanobind.hpp"
#include "ttnn/operations/experimental/dropout/dropout_nanobind.hpp"
#include "ttnn/operations/experimental/bcast_to/bcast_to_nanobind.hpp"
#include "ttnn/operations/experimental/reshape/view_nanobind.hpp"
#include "ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_nanobind.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_nanobind.hpp"
#include "ttnn/operations/experimental/padded_slice/padded_slice_nanobind.hpp"
#include "ttnn/operations/experimental/where/where_nanobind.hpp"
#include "ttnn/operations/experimental/test/hang_device/hang_device_operation_nanobind.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul_nanobind.hpp"

namespace ttnn::operations::experimental {

void py_module(nb::module_& mod) {
    slice_write::bind_slice_write(mod);
    padded_slice::bind_padded_slice(mod);

    transformer::detail::bind_concatenate_heads(mod);
    transformer::detail::bind_split_qkv(mod);
    transformer::detail::bind_nlp_create_qkv_heads(mod);
    transformer::detail::bind_create_qkv_heads_from_separate_tensors(mod);
    nlp_concat_heads_decode::detail::bind_nlp_concat_heads_decode(mod);
    nlp_concat_heads::detail::bind_nlp_concat_heads(mod);
    transformer::detail::bind_nlp_concat_heads_boltz(mod);
    transformer::detail::bind_nlp_create_qkv_heads_decode(mod);
    transformer::detail::bind_nlp_create_qkv_heads_falcon7b(mod);
    transformer::detail::bind_nlp_create_qkv_heads_vit(mod);
    transformer::detail::bind_nlp_create_qkv_heads_segformer(mod);
    transformer::detail::bind_nlp_create_qkv_heads_boltz(mod);
    transformer::detail::bind_nlp_kv_cache_load_slice(mod);
    transformer::detail::bind_all_reduce_create_qkv_heads(mod);

    transformer::bind_wan_fused_distributed_rmsnorm(mod);
    transformer::bind_dit_layernorm_pre_all_gather(mod);
    transformer::bind_dit_layernorm_post_all_gather(mod);
    transformer::bind_rotary_embedding(mod);
    transformer::bind_rotary_embedding_llama(mod);
    transformer::bind_rotary_embedding_llama_fused_qk(mod);
    transformer::bind_rotate_half(mod);

    create_qkv_heads::detail::bind_create_qkv_heads(mod);

    reduction::detail::bind_fast_reduce_nc(mod);
    reduction::detail::bind_reduction_intimg_operation(mod);
    reduction::detail::bind_deepseek_grouped_gate(mod);

    ssm::detail::bind_prefix_scan(mod);
    ssm::detail::bind_repeat_and_interleave_eltwise_mul(mod);
    ssm::detail::bind_hc_sum_reduce(mod);

    cnn::detail::bind_convert_to_chw(mod);
    cnn::detail::bind_convert_to_hwc(mod);

    ttnn::operations::experimental::conv3d::detail::bind_conv3d(mod);
    adaptive_pool::bind_adaptive_avg_pool2d_operation(mod);
    adaptive_pool::bind_adaptive_max_pool2d_operation(mod);

    copy::detail::bind_typecast(mod);

    paged_cache::detail::bind_experimental_paged_cache_operations(mod);
    matmul::detail::bind_attn_matmul(mod);
    matmul::detail::bind_attn_matmul_from_cache(mod);
    matmul::detail::bind_group_attn_matmul(mod);

    plusone::detail::bind_experimental_plusone_operation(mod);
    dropout::detail::bind_experimental_dropout_operation(mod);
    reshape::detail::bind_view(mod);

    gelu_backward::detail::bind_experimental_gelu_backward_operation(mod);

    test::bind_test_hang_device_operation(mod);

    // CCL ops
    auto m_experimental_ccl = mod.def_submodule("ccl_experimental", "experimental collective communication operations");
    ccl::py_module(m_experimental_ccl);

    broadcast_to::detail::bind_broadcast_to(mod);

    operations::experimental::ternary::detail::bind_where(mod);

    minimal_matmul::detail::bind_minimal_matmul(mod);
}

}  // namespace ttnn::operations::experimental
