// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_pybind.hpp"

namespace ttnn::operations::experimental::transformer::detail {

template <typename transformer_operation_t>
void bind_nlp_create_qkv_heads_template(pybind11::module& module, const transformer_operation_t& operation) {
    ttnn::bind_registered_operation(
        module,
        operation,
        R"doc(
             Shuffles [B, 1, S, 3 * head_dim * num_heads] fused qkv matrix into 3 Q, K, and V heads with shapes [B, num_heads, S, head_dim], [B, num_kv_heads, head_dim, S], and [B, num_kv_heads, S, head_dim]. If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and ``input`` should have shape [B, 1, S, head_dim * num_heads] instead. ``num_kv_heads`` defaults to ``num_heads`` if not provided. An additional transpose along the last two dims is performed by default for K heads, but this can be skipped with ``transpose_k_heads=false``.
        )doc",
        ttnn::pybind_overload_t{[](const transformer_operation_t& self,
                                   const ttnn::Tensor& input_tensor_q,
                                   const std::optional<ttnn::Tensor>& input_tensor_kv,
                                   const uint32_t num_heads,
                                   const std::optional<uint32_t> num_kv_heads,
                                   const bool transpose_k_heads,
                                   const std::optional<ttnn::MemoryConfig>& memory_config,
                                   std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors,
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
                                pybind11::arg("input_kv").noconvert() = std::nullopt,
                                pybind11::kw_only(),
                                pybind11::arg("num_heads").noconvert(),
                                pybind11::arg("num_kv_heads").noconvert() = std::nullopt,
                                pybind11::arg("transpose_k_heads").noconvert() = true,
                                pybind11::arg("memory_config").noconvert() = std::nullopt,
                                pybind11::arg("output_tensors").noconvert() = std::nullopt,
                                pybind11::arg("queue_id") = 0});
};

void bind_nlp_create_qkv_heads(pybind11::module& module) {
    bind_nlp_create_qkv_heads_template(module, ttnn::experimental::nlp_create_qkv_heads);
}
}  // namespace ttnn::operations::experimental::transformer::detail
