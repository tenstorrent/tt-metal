// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "softmax.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_softmax(nb::module_& mod) {
    const auto* doc = R"doc(
        Softmax over the last dimension of :attr:`input_tensor` (Quasar / Metal 2.0).
        Same contract as :func:`ttnn.softmax` with ``dim=-1``; see that op for the full description.

        Args:
            input_tensor (ttnn.Tensor): the input tensor (TILE layout, interleaved, bfloat16).

        Keyword args:
            dim (int): must resolve to the last dimension. Defaults to `-1`.
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            numeric_stable (bool): subtract the row max before the exponent. Defaults to `True`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"softmax", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            int,
            const std::optional<ttnn::MemoryConfig>&,
            std::optional<const ttnn::DeviceComputeKernelConfig>,
            bool>(&ttnn::operations::experimental::quasar::softmax),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("dim") = -1,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("numeric_stable") = true);
}

}  // namespace ttnn::operations::experimental::quasar::detail
