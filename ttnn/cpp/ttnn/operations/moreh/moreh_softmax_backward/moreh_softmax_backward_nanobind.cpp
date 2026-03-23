// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_softmax_backward.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

void bind_moreh_softmax_backward_operation(nb::module_& mod) {
    export_enum<MorehSoftmaxBackwardOp>(mod, "MorehSoftmaxBackwardOp");
    export_enum<MorehSoftmaxBackwardOpParallelizationStrategy>(mod, "MorehSoftmaxBackwardOpParallelizationStrategy");

    ttnn::bind_function<"moreh_softmax_backward">(
        mod,
        "Moreh Softmax Backward Operation",
        ttnn::overload_t(
            &ttnn::moreh_softmax_backward,
            nb::arg("output_tensor"),
            nb::arg("output_grad_tensor"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("input_grad_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxBackwardOp::SOFTMAX,
            nb::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    ttnn::bind_function<"moreh_softmin_backward">(
        mod,
        "Moreh Softmin Backward Operation",
        ttnn::overload_t(
            &ttnn::moreh_softmin_backward,
            nb::arg("output_tensor"),
            nb::arg("output_grad_tensor"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("input_grad_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxBackwardOp::SOFTMIN,
            nb::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    ttnn::bind_function<"moreh_logsoftmax_backward">(
        mod,
        "Moreh LogSoftmax Backward Operation",
        ttnn::overload_t(
            &ttnn::moreh_logsoftmax_backward,
            nb::arg("output_tensor"),
            nb::arg("output_grad_tensor"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("input_grad_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxBackwardOp::LOGSOFTMAX,
            nb::arg("strategy") = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
