// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_kv_cache_load_slice(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::nlp_kv_cache_load_slice);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_kv_cache_load_slice,
        R"doc(
            Unpad TT INTERLEAVED, TILE layout Tensor into a height sharded tensor. Typically used to unpad the KV cache from [B,n_heads,max_seq_length,head_dim] (or [n_heads,B,max_seq_length,head_dim]) into [B,n_heads,S,head_dim] (or [n_heads,B,S,head_dim]), where S = seq_len_end-seq_len_start. seq_len_start and seq_len_end are the start and end of the sequence length to unpad, and must be multiples of 32.
            Returns an output tensor that is height sharded on B x n_heads corees (note the B and n_heads dims are interchangeable), where each shard is [S, head_dim].
        )doc",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t seq_len_start,
               const uint32_t seq_len_end,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor,
               uint8_t queue_id) {
                return self(queue_id, input_tensor, seq_len_start, seq_len_end, memory_config, optional_output_tensor);
            },
            pybind11::arg("input_tensor").noconvert(),
            pybind11::kw_only(),
            pybind11::arg("seq_len_start").noconvert(),
            pybind11::arg("seq_len_end").noconvert(),
            pybind11::arg("memory_config") = std::nullopt,
            pybind11::arg("output_tensor") = std::nullopt,
            pybind11::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::transformer::detail
