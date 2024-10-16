// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_pybind.hpp"

#include "moreh_softmax_backward.hpp"
#include "pybind11/decorators.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

void bind_moreh_softmax_backward_operation(py::module& module) {
    export_enum<MorehSoftmaxBackwardOp>(module, "MorehSoftmaxBackwardOp");
    export_enum<MorehSoftmaxBackwardOpParallelizationStrategy>(module, "MorehSoftmaxBackwardOpParallelizationStrategy");

#define BIND_MOREH_SOFT_BACKWARD_OP(op_name, op_enum, op_desc)                                              \
    bind_registered_operation(                                                                              \
        module,                                                                                             \
        ttnn::op_name,                                                                                      \
        op_desc,                                                                                            \
        ttnn::pybind_arguments_t{py::arg("output_tensor"),                                                  \
                                 py::arg("output_grad_tensor"),                                             \
                                 py::arg("dim"),                                                            \
                                 py::kw_only(),                                                             \
                                 py::arg("input_grad_tensor") = std::nullopt,                               \
                                 py::arg("op") = op_enum,                                                   \
                                 py::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE, \
                                 py::arg("memory_config") = std::nullopt,                                   \
                                 py::arg("compute_kernel_config") = std::nullopt});

    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_softmax_backward, MorehSoftmaxBackwardOp::SOFTMAX, "Moreh Softmax Backward Operation")
    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_softmin_backward, MorehSoftmaxBackwardOp::SOFTMIN, "Moreh Softmin Backward Operation")
    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_logsoftmax_backward, MorehSoftmaxBackwardOp::LOGSOFTMAX, "Moreh LogSoftmax Backward Operation")
#undef BIND_MOREH_SOFT_BACKWARD_OP
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
