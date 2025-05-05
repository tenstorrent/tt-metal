// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_pybind.hpp"

#include "moreh_softmax.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn-pybind/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

void bind_moreh_softmax_operation(py::module& module) {
    export_enum<MorehSoftmaxOp>(module, "MorehSoftmaxOp");
    export_enum<MorehSoftmaxOpParallelizationStrategy>(module, "MorehSoftmaxOpParallelizationStrategy");

#define BIND_MOREH_SOFT_OP(op_name, op_enum, op_desc)                          \
    bind_registered_operation(                                                 \
        module,                                                                \
        ttnn::op_name,                                                         \
        op_desc,                                                               \
        ttnn::pybind_arguments_t{                                              \
            py::arg("input_tensor"),                                           \
            py::arg("dim"),                                                    \
            py::kw_only(),                                                     \
            py::arg("output_tensor") = std::nullopt,                           \
            py::arg("op") = op_enum,                                           \
            py::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE, \
            py::arg("memory_config") = std::nullopt,                           \
            py::arg("compute_kernel_config") = std::nullopt});

    BIND_MOREH_SOFT_OP(moreh_softmax, MorehSoftmaxOp::SOFTMAX, "Moreh Softmax Operation")
    BIND_MOREH_SOFT_OP(moreh_softmin, MorehSoftmaxOp::SOFTMIN, "Moreh Softmin Operation")
    BIND_MOREH_SOFT_OP(moreh_logsoftmax, MorehSoftmaxOp::LOGSOFTMAX, "Moreh LogSoftmax Operation")
#undef BIND_MOREH_SOFT_OP
}

}  // namespace ttnn::operations::moreh::moreh_softmax
