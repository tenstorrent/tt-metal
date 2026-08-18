// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_nanobind.hpp"
#include "sigmoid_gated_rms_norm.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail {

void bind_sigmoid_gated_rms_norm(nb::module_& mod) {
    ttnn::bind_function<"sigmoid_gated_rms_norm", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply per-head RMS normalization followed by sigmoid gating.

        For input head ``h``:

            normalized = input / sqrt(mean(input², dim=V) + epsilon)
            output = normalized * weight * sigmoid(gate)

        The operation converts head-first input ``[B*H, T, V]`` into time-first
        output ``[B, T, H*V]`` for the following output projection.

        Args:
            input (ttnn.Tensor): Input tensor ``[B*H, T, V]``. Must be an
                interleaved TILE-layout device tensor with FLOAT32 or BFLOAT16 dtype.
            gate (ttnn.Tensor): Sigmoid gate ``[B, T, H*V]``. Must be an
                interleaved TILE-layout BFLOAT16 device tensor.
            weight (ttnn.Tensor): Per-value RMSNorm weight ``[V]``. Must be an
                interleaved TILE-layout BFLOAT16 device tensor.
            num_heads (int): Number of heads ``H``. The input leading dimension
                must be divisible by ``H``.

        Keyword Args:
            epsilon (float): Finite positive RMSNorm epsilon. Defaults to ``1e-5``.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.
            output_dtype (ttnn.DataType): Output dtype, either FLOAT32 or BFLOAT16.
                Defaults to FLOAT32.

        Returns:
            ttnn.Tensor: A new TILE-layout tensor with shape ``[B, T, H*V]``.

        Note:
            ``T`` and ``V`` must be positive and tile-aligned. All input tensors
            must be allocated on the same device. Inputs are not modified.
        )doc",
        &ttnn::experimental::kda::sigmoid_gated_rms_norm,
        nb::arg("input").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::FLOAT32);
}

}  // namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail
