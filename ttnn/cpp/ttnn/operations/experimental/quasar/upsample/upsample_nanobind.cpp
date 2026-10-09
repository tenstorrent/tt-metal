// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/upsample/upsample.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_upsample(nb::module_& mod) {
    const auto* doc = R"doc(
        Nearest-neighbour upsample of [N, H, W, C] data with integer scale factors (Quasar / Metal 2.0).

        Supports row-major height / block sharded inputs and interleaved (row-major or tiled) inputs.
        Bilinear mode and fractional scale factors are not ported.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale_factor (int or [int, int]): multiplier for spatial size, uniform or [H, W].

        Keyword args:
            mode (str, optional): only 'nearest'. Defaults to 'nearest'.
            memory_config (ttnn.MemoryConfig, optional): output memory configuration. Defaults to the input's.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): unused by the nearest path.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"upsample", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::upsample,
        nb::arg("input_tensor"),
        nb::arg("scale_factor"),
        nb::kw_only(),
        nb::arg("mode") = "nearest",
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
