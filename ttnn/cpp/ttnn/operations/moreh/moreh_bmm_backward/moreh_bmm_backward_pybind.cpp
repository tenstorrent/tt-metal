// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
void bind_moreh_bmm_backward_operation(py::module& module) {
    bind_registered_operation(module,
                              ttnn::moreh_bmm_backward,
                              "Moreh BMM Backward Operation",
                              ttnn::pybind_arguments_t{py::arg("output_grad"),
                                                       py::arg("input"),
                                                       py::arg("mat2"),
                                                       py::kw_only(),
                                                       py::arg("are_required_outputs") = std::vector<bool>{true, true},
                                                       py::arg("input_grad") = std::nullopt,
                                                       py::arg("mat2_grad") = std::nullopt,
                                                       py::arg("input_grad_memory_config") = std::nullopt,
                                                       py::arg("mat2_grad_memory_config") = std::nullopt,
                                                       py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
