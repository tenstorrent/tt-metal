// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_softmax_backward.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "cpp/ttnn-nanobind/export_enum.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_softmax_backward {

void bind_moreh_softmax_backward_operation(nb::module_& mod) {
    export_enum<MorehSoftmaxBackwardOp>(mod, "MorehSoftmaxBackwardOp");
    export_enum<MorehSoftmaxBackwardOpParallelizationStrategy>(mod, "MorehSoftmaxBackwardOpParallelizationStrategy");

#define BIND_MOREH_SOFT_BACKWARD_OP(op_name, op_enum, op_desc)                         \
    bind_registered_operation(                                                         \
        mod,                                                                           \
        ttnn::op_name,                                                                 \
        op_desc,                                                                       \
        ttnn::nanobind_arguments_t{                                                    \
            nb::arg("output_tensor"),                                                  \
            nb::arg("output_grad_tensor"),                                             \
            nb::arg("dim"),                                                            \
            nb::kw_only(),                                                             \
            nb::arg("input_grad_tensor") = std::nullopt,                               \
            nb::arg("op") = op_enum,                                                   \
            nb::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE, \
            nb::arg("memory_config") = std::nullopt,                                   \
            nb::arg("compute_kernel_config") = std::nullopt});

    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_softmax_backward, MorehSoftmaxBackwardOp::SOFTMAX, "Moreh Softmax Backward Operation")
    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_softmin_backward, MorehSoftmaxBackwardOp::SOFTMIN, "Moreh Softmin Backward Operation")
    BIND_MOREH_SOFT_BACKWARD_OP(
        moreh_logsoftmax_backward, MorehSoftmaxBackwardOp::LOGSOFTMAX, "Moreh LogSoftmax Backward Operation")
#undef BIND_MOREH_SOFT_BACKWARD_OP
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
