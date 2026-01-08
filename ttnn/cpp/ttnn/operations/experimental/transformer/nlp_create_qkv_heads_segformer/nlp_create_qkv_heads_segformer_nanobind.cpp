// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_segformer_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_segformer(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::nlp_create_qkv_heads_segformer,
        R"doc(
            Shuffles [B, 1, S, 2304] fused qkv matrix into 3 heads with shapes [B, 12, S, 64], [B, 12, S, 64], and [B, 12, S, 64].
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::nlp_create_qkv_heads_segformer)& self,
               const ttnn::Tensor& input_tensor_q,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors) {
                return self(input_tensor_q, memory_config, optional_output_tensors);
            },
            nb::arg("input").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("output_tensors").noconvert() = nb::none()});
};
}  // namespace ttnn::operations::experimental::transformer::detail
