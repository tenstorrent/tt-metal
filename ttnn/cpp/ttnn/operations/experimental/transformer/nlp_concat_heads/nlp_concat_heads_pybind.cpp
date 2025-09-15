// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"

namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;

void bind_nlp_concat_heads(py::module& module) {
    using OperationType = decltype(ttnn::experimental::nlp_concat_heads);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_concat_heads,
        R"doc(
            Shuffles [B, num_heads, S, head_dim] tensor into tensor with shape [B, 1, S, num_heads * head_dim].
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer::detail
