// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {
void bind_moreh_softmax_backward_operation(py::module& module) {
    export_enum<MorehSoftmaxBackwardOpParallelizationStrategy>(module, "MorehSoftmaxBackwardOpParallelizationStrategy");
    export_enum<MorehSoftmaxBackwardOp>(module, "MorehSoftmaxBackwardOp");

    bind_registered_operation(
        module,
        ttnn::moreh_softmax_backward,
        "Moreh moreh_softmax_backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_tensor"),
            py::arg("output_grad_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("input_grad_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxBackwardOp::SOFTMAX,
            py::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::moreh_softmin_backward,
        "Moreh moreh_softmin Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_tensor"),
            py::arg("output_grad_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("input_grad_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxBackwardOp::SOFTMIN,
            py::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::moreh_logsoftmax_backward,
        "Moreh moreh_logsoftmax Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_tensor"),
            py::arg("output_grad_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("input_grad_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxBackwardOp::LOGSOFTMAX,
            py::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}
