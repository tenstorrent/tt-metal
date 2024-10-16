// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_pybind.hpp"

namespace ttnn::operations::experimental::transformer::detail {

template <typename transformer_operation_t>
void bind_create_qkv_heads_from_separate_tensors_template(pybind11::module& module,
                                                          const transformer_operation_t& operation) {
    ttnn::bind_registered_operation(
        module,
        operation,
        R"doc(
            Splits a [B, 1, Seq_len, H] q matrix and fused kv matrix (where H is num_q_heads * head_dim for q and num_kv_heads * head_dim * 2 for kv) into a Q tensor [B, num_q_heads, Seq_len, head_dim], K tensor [B, num_kv_heads, Seq_len, head_dim] (with the last two dims transposed if applicable) and V tensor [B, num_kv_heads, Seq_len, head_dim].
        )doc",
        ttnn::pybind_overload_t{[](const transformer_operation_t& self,
                                   const ttnn::Tensor& input_tensor_q,
                                   const ttnn::Tensor& input_tensor_kv,
                                   const uint32_t num_heads,
                                   const std::optional<uint32_t> num_kv_heads,
                                   const bool transpose_k_heads,
                                   const std::optional<ttnn::MemoryConfig>& memory_config,
                                   std::optional<std::array<Tensor, 3>> optional_output_tensors,
                                   uint8_t queue_id) {
                                    return self(queue_id,
                                                input_tensor_q,
                                                input_tensor_kv,
                                                num_heads,
                                                num_kv_heads,
                                                transpose_k_heads,
                                                memory_config,
                                                optional_output_tensors);
                                },
                                pybind11::arg("input").noconvert(),
                                pybind11::arg("input_kv").noconvert(),
                                pybind11::kw_only(),
                                pybind11::arg("num_heads").noconvert(),
                                pybind11::arg("num_kv_heads").noconvert() = std::nullopt,
                                pybind11::arg("transpose_k_heads").noconvert() = true,
                                pybind11::arg("memory_config").noconvert() = std::nullopt,
                                pybind11::arg("output_tensors").noconvert() = std::nullopt,
                                pybind11::arg("queue_id") = 0});
};

void bind_create_qkv_heads_from_separate_tensors(pybind11::module& module) {
    bind_create_qkv_heads_from_separate_tensors_template(module,
                                                         ttnn::experimental::create_qkv_heads_from_separate_tensors);
}
}  // namespace ttnn::operations::experimental::transformer::detail
