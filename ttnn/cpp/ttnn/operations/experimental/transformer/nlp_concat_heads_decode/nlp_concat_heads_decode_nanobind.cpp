// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_decode_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_concat_heads_decode(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::nlp_concat_heads_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::nlp_concat_heads_decode,
        R"doc(
            Shuffles [S=1, B=32, 32(num_heads), head_dim] tensor into tensor with shape [S=1, 1, B=32, num_heads * head_dim]. num_heads should be specified and be less than 32; the op will assume the input padded num_heads to 32 and will unpad it. The output is default width sharded by num heads.
        )doc",
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t num_heads,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor) {
                return self(input_tensor, num_heads, memory_config, optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("num_heads").noconvert(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer::detail
