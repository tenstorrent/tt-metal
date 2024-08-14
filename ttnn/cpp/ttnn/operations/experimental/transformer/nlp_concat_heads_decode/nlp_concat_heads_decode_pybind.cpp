// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"


namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;

void bind_nlp_concat_heads_decode(py::module& module) {
    using OperationType = decltype(ttnn::experimental::nlp_concat_heads_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_concat_heads_decode,
        R"doc(
            Shuffles [S=1, B=32, 32(num_heads), head_dim] tensor into tensor with shape [S=1, 1, B=32, num_heads * head_dim]. num_heads should be specified and be less than 32; the op will assume the input padded num_heads to 32 and will unpad it. The output is default width sharded by num heads.
        )doc",
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const uint32_t num_heads,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                std::optional<ttnn::Tensor> optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, num_heads, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("num_heads").noconvert(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0});

}

}  // namespace ttnn::operations::experimental::transformer::detail
