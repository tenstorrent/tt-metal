// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_depthwise_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "conv1d_depthwise.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::conv1d_depthwise::detail {

void bind_conv1d_depthwise(nb::module_& mod) {
    ttnn::bind_function<"conv1d_depthwise", "ttnn.experimental.">(
        mod,
        R"doc(
        Depthwise 1D FIR filter with taps shared across all channels:
            y[b, t, c] = sum_{j<K} taps[j] * x[b, t*stride + j, c]

        Input and output are (B, T_pad, C) ROW_MAJOR FLOAT32 interleaved tensors.
        Output length is T_out = (T_pad - len(taps)) / stride + 1. The input must be
        pre-padded by the caller; any cross-shard halo is supplied upstream.

        Args:
            input_tensor (ttnn.Tensor): (B, T_pad, C) ROW_MAJOR FLOAT32 input.
            taps (List[float]): K filter taps, shared across all channels.
            stride (int): output stride.

        Keyword Args:
            dtype (ttnn.DataType, optional): output dtype (defaults to input dtype).
            compute_config (ttnn.DeviceComputeKernelConfig, optional): compute kernel config.
            memory_config (ttnn.MemoryConfig, optional): output memory config.

        Returns:
            ttnn.Tensor: (B, T_out, C) ROW_MAJOR FLOAT32 output.
        )doc",
        &ttnn::experimental::conv1d_depthwise,
        nb::arg("input_tensor"),
        nb::arg("taps"),
        nb::arg("stride") = 1,
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::conv1d_depthwise::detail
