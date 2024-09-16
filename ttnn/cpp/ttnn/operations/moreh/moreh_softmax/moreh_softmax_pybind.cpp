// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax {
void bind_moreh_softmax_operation(py::module& module) {
    export_enum<MorehSoftmaxOpParallelizationStrategy>(module, "MorehSoftmaxOpParallelizationStrategy");
    export_enum<MorehSoftmaxOp>(module, "MorehSoftmaxOp");

    bind_registered_operation(
        module,
        ttnn::moreh_softmax,
        "Moreh moreh_softmax Operation",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxOp::SOFTMAX,
            py::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::moreh_softmin,
        "Moreh moreh_softmin Operation",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxOp::SOFTMIN,
            py::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::moreh_logsoftmax,
        "Moreh moreh_logsoftmax Operation",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("op") = MorehSoftmaxOp::LOGSOFTMAX,
            py::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            py::arg("output_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}
