// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "batch_norm.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_batch_norm(nb::module_& mod) {
    const auto* doc = R"doc(
        Batch normalization of :attr:`input` in inference mode (Quasar / Metal 2.0). Same contract as
        :func:`ttnn.batch_norm` with ``training=False``; see that op for the full description.

        Args:
            input (ttnn.Tensor): the [N, C, H, W] input tensor (TILE layout, interleaved, bfloat16).

        Keyword args:
            running_mean (ttnn.Tensor): [1, C, 1, 1] per-channel mean. Required.
            running_var (ttnn.Tensor): [1, C, 1, 1] per-channel variance. Required.
            training (bool): must be `False`. Defaults to `False`.
            eps (float): Defaults to `1e-05`.
            momentum (float): unused in inference. Defaults to `0.1`.
            weight (ttnn.Tensor, optional): [1, C, 1, 1] per-channel scale. Defaults to `None`.
            bias (ttnn.Tensor, optional): [1, C, 1, 1] per-channel shift. Defaults to `None`.
            output (ttnn.Tensor, optional): not supported. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"batch_norm", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            std::optional<ttnn::Tensor>,
            std::optional<ttnn::Tensor>,
            bool,
            float,
            float,
            const std::optional<ttnn::Tensor>&,
            const std::optional<ttnn::Tensor>&,
            const std::optional<ttnn::Tensor>&,
            const std::optional<ttnn::MemoryConfig>&,
            const std::optional<ttnn::DeviceComputeKernelConfig>&>(&ttnn::operations::experimental::quasar::batch_norm),
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("running_mean") = nb::none(),
        nb::arg("running_var") = nb::none(),
        nb::arg("training") = false,
        nb::arg("eps") = 1e-05,
        nb::arg("momentum") = 0.1,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
