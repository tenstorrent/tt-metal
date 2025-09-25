// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_boltz_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/nlp_create_qkv_heads_boltz.hpp"

namespace ttnn::operations::experimental::transformer::detail {
template <typename transformer_operation_t>
void bind_nlp_create_qkv_heads_boltz_template(nb::module_& mod, const transformer_operation_t& operation) {
    ttnn::bind_registered_operation(
        mod,
        operation,
        R"doc(
             Shuffles [B, 1, S, 3 * head_dim * num_heads] fused qkv matrix into 3 Q, K, and V heads with shapes [B, num_heads, S, head_dim], [B, num_kv_heads, head_dim, S], and [B, num_kv_heads, S, head_dim]. If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and ``input`` should have shape [B, 1, S, head_dim * num_heads] instead. ``num_kv_heads`` defaults to ``num_heads`` if not provided. An additional transpose along the last two dims is performed by default for K heads, but this can be skipped with ``transpose_k_heads=false``.
        )doc",
        ttnn::nanobind_overload_t{
            [](const transformer_operation_t& self,
               const ttnn::Tensor& input_tensor_q,
               const std::optional<ttnn::Tensor>& input_tensor_kv,
               const uint32_t num_heads,
               const std::optional<uint32_t> num_kv_heads,
               const bool transpose_k_heads,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors) {
                return self(
                    input_tensor_q,
                    input_tensor_kv,
                    num_heads,
                    num_kv_heads,
                    transpose_k_heads,
                    memory_config,
                    optional_output_tensors);
            },
            nb::arg("input").noconvert(),
            nb::arg("input_kv").noconvert() = nb::none(),
            nb::kw_only(),
            nb::arg("num_heads").noconvert(),
            nb::arg("num_kv_heads").noconvert() = nb::none(),
            nb::arg("transpose_k_heads").noconvert() = true,
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("output_tensors").noconvert() = nb::none()});
};

void bind_nlp_create_qkv_heads_boltz(nb::module_& mod) {
    bind_nlp_create_qkv_heads_boltz_template(mod, ttnn::experimental::nlp_create_qkv_heads_boltz);
}
}  // namespace ttnn::operations::experimental::transformer::detail
