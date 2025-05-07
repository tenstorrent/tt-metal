// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/examples/example_multiple_return/example_multiple_return_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"

namespace py = pybind11;

namespace ttnn::operations::examples {

void bind_example_multiple_return_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::prim::example_multiple_return,
        R"doc(example_multiple_return(input_tensor: ttnn.Tensor) -> std::vector<std::optional<ttnn.Tensor>>)doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"), py::arg("return_output1"), py::arg("return_output2")});

    bind_registered_operation(
        module,
        ttnn::composite_example_multiple_return,
        R"doc(composite_example_multiple_return(input_tensor: ttnn.Tensor) -> std::vector<std::optional<Tensor>>)doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"), py::arg("return_output1"), py::arg("return_output2")});
}

}  // namespace ttnn::operations::examples
