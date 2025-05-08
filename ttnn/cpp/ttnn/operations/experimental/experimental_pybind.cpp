// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_pybind.hpp"
#include "ttnn/operations/experimental/conv3d/conv3d_pybind.hpp"
#include "ttnn/operations/experimental/reduction/argmax/argmax_pybind.hpp"
#include "ttnn/operations/experimental/reduction/cumprod/cumprod_pybind.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/cumsum_pybind.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_pybind.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write_pybind.hpp"
#include "ttnn/operations/experimental/ssm/hc_sum_reduce/hc_sum_reduce_pybind.hpp"
#include "ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_pybind.hpp"
#include "ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_pybind.hpp"
#include "ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads_pybind.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads_pybind.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice_pybind.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache_pybind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_pybind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_pybind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk_pybind.hpp"
#include "ttnn/operations/experimental/transformer/rotate_half/rotate_half_pybind.hpp"
#include "ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_pybind.hpp"
#include "cpp/ttnn/operations/experimental/copy/typecast/typecast_pybind.hpp"
#include "cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_pybind.hpp"
#include "cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_pybind.hpp"
#include "ttnn/operations/experimental/ccl/ccl_experimental_pybind.hpp"
#include "ttnn/operations/experimental/plusone/plusone_pybind.hpp"
#include "ttnn/operations/experimental/dropout/dropout_pybind.hpp"
#include "ttnn/operations/experimental/bcast_to/bcast_to_pybind.hpp"

#include "ttnn/operations/experimental/reshape/view_pybind.hpp"
#include "ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_pybind.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_pybind.hpp"
#include "ttnn/operations/experimental/reduction/sort/sort_pybind.hpp"

namespace ttnn::operations::experimental {

void py_module(py::module& module) {
    slice_write::bind_slice_write(module);

    transformer::detail::bind_concatenate_heads(module);
    transformer::detail::bind_split_qkv(module);
    transformer::detail::bind_nlp_create_qkv_heads(module);
    transformer::detail::bind_create_qkv_heads(module);
    transformer::detail::bind_create_qkv_heads_from_separate_tensors(module);
    transformer::detail::bind_nlp_concat_heads(module);
    transformer::detail::bind_nlp_concat_heads_decode(module);
    transformer::detail::bind_nlp_create_qkv_heads_decode(module);
    transformer::detail::bind_nlp_create_qkv_heads_falcon7b(module);
    transformer::detail::bind_nlp_create_qkv_heads_vit(module);
    transformer::detail::bind_nlp_create_qkv_heads_segformer(module);
    transformer::detail::bind_nlp_kv_cache_load_slice(module);
    transformer::detail::py_bind_all_reduce_create_qkv_heads(module);

    transformer::py_bind_rotary_embedding(module);
    transformer::py_bind_rotary_embedding_llama(module);
    transformer::py_bind_rotary_embedding_llama_fused_qk(module);
    transformer::py_bind_rotate_half(module);

    reduction::detail::bind_argmax_operation(module);
    reduction::detail::bind_argmin_operation(module);
    reduction::detail::bind_fast_reduce_nc(module);

    ssm::detail::bind_prefix_scan(module);
    ssm::detail::bind_repeat_and_interleave_eltwise_mul(module);
    ssm::detail::bind_hc_sum_reduce(module);

    cnn::detail::bind_convert_to_chw(module);

    ttnn::operations::experimental::conv3d::detail::py_bind_conv3d(module);
    ttnn::operations::experimental::reduction::cumprod::detail::bind_cumprod_operation(module);

    copy::detail::py_bind_typecast(module);

    paged_cache::detail::bind_experimental_paged_cache_operations(module);
    matmul::detail::bind_attn_matmul(module);
    matmul::detail::bind_attn_matmul_from_cache(module);
    matmul::detail::bind_group_attn_matmul(module);

    plusone::detail::bind_experimental_plusone_operation(module);
    dropout::detail::bind_experimental_dropout_operation(module);
    reshape::detail::py_bind_view(module);

    gelu_backward::detail::bind_experimental_gelu_backward_operation(module);

    reduction::sort::detail::bind_reduction_sort_operation(module);

    reduction::detail::bind_cumsum_operation(module);

    // CCL ops
    auto m_experimental_ccl =
        module.def_submodule("ccl_experimental", "experimental collective communication operations");
    ccl::py_module(m_experimental_ccl);

    broadcast_to::detail::py_bind_broadcast_to(module);
}

}  // namespace ttnn::operations::experimental
