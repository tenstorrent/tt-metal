// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_softmax.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

void bind_moreh_softmax_operation(nb::module_& mod) {
    export_enum<MorehSoftmaxOp>(mod, "MorehSoftmaxOp");
    export_enum<MorehSoftmaxOpParallelizationStrategy>(mod, "MorehSoftmaxOpParallelizationStrategy");

    const auto* moreh_softmax_doc =
        R"doc(
        Moreh Softmax Operation

        Computes softmax along the specified dimension.

        Args:
            input_tensor (ttnn.Tensor): The input tensor.
            dim (int): The dimension along which to compute softmax.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to None.
            op (MorehSoftmaxOp, optional): The softmax operation type. Defaults to MorehSoftmaxOp.SOFTMAX.
            strategy (MorehSoftmaxOpParallelizationStrategy, optional): Parallelization strategy. Defaults to MorehSoftmaxOpParallelizationStrategy.NONE.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to None.

        Returns:
            ttnn.Tensor: The output tensor with softmax applied.
        )doc";

    ttnn::bind_function<"moreh_softmax">(
        mod,
        moreh_softmax_doc,
        ttnn::overload_t(
            &ttnn::moreh_softmax,
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim").noconvert(),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxOp::SOFTMAX,
            nb::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    const auto* moreh_softmin_doc =
        R"doc(
        Moreh Softmin Operation

        Computes softmin along the specified dimension.

        Args:
            input_tensor (ttnn.Tensor): The input tensor.
            dim (int): The dimension along which to compute softmin.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to None.
            op (MorehSoftmaxOp, optional): The softmax operation type. Defaults to MorehSoftmaxOp.SOFTMIN.
            strategy (MorehSoftmaxOpParallelizationStrategy, optional): Parallelization strategy. Defaults to MorehSoftmaxOpParallelizationStrategy.NONE.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to None.

        Returns:
            ttnn.Tensor: The output tensor with softmin applied.
        )doc";

    ttnn::bind_function<"moreh_softmin">(
        mod,
        moreh_softmin_doc,
        ttnn::overload_t(
            &ttnn::moreh_softmin,
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim").noconvert(),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxOp::SOFTMIN,
            nb::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    const auto* moreh_logsoftmax_doc =
        R"doc(
        Moreh LogSoftmax Operation

        Computes log-softmax along the specified dimension.

        Args:
            input_tensor (ttnn.Tensor): The input tensor.
            dim (int): The dimension along which to compute log-softmax.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to None.
            op (MorehSoftmaxOp, optional): The softmax operation type. Defaults to MorehSoftmaxOp.LOGSOFTMAX.
            strategy (MorehSoftmaxOpParallelizationStrategy, optional): Parallelization strategy. Defaults to MorehSoftmaxOpParallelizationStrategy.NONE.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to None.

        Returns:
            ttnn.Tensor: The output tensor with log-softmax applied.
        )doc";

    ttnn::bind_function<"moreh_logsoftmax">(
        mod,
        moreh_logsoftmax_doc,
        ttnn::overload_t(
            &ttnn::moreh_logsoftmax,
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim").noconvert(),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("op") = MorehSoftmaxOp::LOGSOFTMAX,
            nb::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_softmax
