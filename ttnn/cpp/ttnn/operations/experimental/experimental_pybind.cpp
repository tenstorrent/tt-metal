// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_pybind.hpp"
#include "ttnn/operations/experimental/reduction/argmax/argmax_pybind.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_pybind.hpp"
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
#include "ttnn/cpp/ttnn/operations/experimental/copy/typecast/typecast_pybind.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_pybind.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_pybind.hpp"
#include "ttnn/operations/experimental/ccl/ccl_experimental_pybind.hpp"
#include "ttnn/operations/experimental/plusone/plusone_pybind.hpp"
#include "ttnn/operations/experimental/dropout/dropout_pybind.hpp"
#include "ttnn/operations/experimental/transformer/speculative_sdpa_decode/speculative_sdpa_decode_pybind.hpp"
namespace ttnn::operations::experimental {

void py_module(py::module& module) {
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

    copy::detail::py_bind_typecast(module);

    paged_cache::detail::bind_experimental_paged_cache_operations(module);
    matmul::detail::bind_attn_matmul(module);
    matmul::detail::bind_attn_matmul_from_cache(module);
    matmul::detail::bind_group_attn_matmul(module);

    plusone::detail::bind_experimental_plusone_operation(module);
    dropout::detail::bind_experimental_dropout_operation(module);
    // CCL ops
    auto m_experimental_ccl =
        module.def_submodule("ccl_experimental", "experimental collective communication operations");
    ccl::py_module(m_experimental_ccl);

    transformer::py_bind_speculative_sdpa_decode(module);
}

}  // namespace ttnn::operations::experimental
