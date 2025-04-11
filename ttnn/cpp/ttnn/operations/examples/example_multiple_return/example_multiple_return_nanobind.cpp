// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/examples/example_multiple_return/example_multiple_return_nanobind.hpp"

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"

namespace nb = nanobind;

namespace ttnn::operations::examples {

void bind_example_multiple_return_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::prim::example_multiple_return,
        R"doc(example_multiple_return(input_tensor: ttnn.Tensor) -> std::vector<std::optional<ttnn.Tensor>>)doc",
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("return_output1"), nb::arg("return_output2")});

    bind_registered_operation(
        mod,
        ttnn::composite_example_multiple_return,
        R"doc(composite_example_multiple_return(input_tensor: ttnn.Tensor) -> std::vector<std::optional<Tensor>>)doc",
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("return_output1"), nb::arg("return_output2")});
}

}  // namespace ttnn::operations::examples
