// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/decorators.hpp"
#include "create_qkv_heads_from_separate_tensors.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::transformer::detail {

namespace {
template <typename transformer_operation_t>
void bind_create_qkv_heads_from_separate_tensors_template(nb::module_& mod, const transformer_operation_t& operation) {
    ttnn::bind_registered_operation(
        mod,
        operation,
        R"doc(
            Splits a [B, 1, Seq_len, H] q matrix and fused kv matrix (where H is num_q_heads * head_dim for q and num_kv_heads * head_dim * 2 for kv) into a Q tensor [B, num_q_heads, Seq_len, head_dim], K tensor [B, num_kv_heads, Seq_len, head_dim] (with the last two dims transposed if applicable) and V tensor [B, num_kv_heads, Seq_len, head_dim].
        )doc",
        ttnn::nanobind_overload_t{
            [](const transformer_operation_t& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_kv,
               const uint32_t num_heads,
               const std::optional<uint32_t> num_kv_heads,
               const bool transpose_k_heads,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<std::array<Tensor, 3>> optional_output_tensors) {
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
            nb::arg("input_kv").noconvert(),
            nb::kw_only(),
            nb::arg("num_heads").noconvert(),
            nb::arg("num_kv_heads").noconvert() = nb::none(),
            nb::arg("transpose_k_heads").noconvert() = true,
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("output_tensors").noconvert() = nb::none()});
};

}  // namespace

void bind_create_qkv_heads_from_separate_tensors(nb::module_& mod) {
    bind_create_qkv_heads_from_separate_tensors_template(
        mod, ttnn::experimental::create_qkv_heads_from_separate_tensors);
}
}  // namespace ttnn::operations::experimental::transformer::detail
