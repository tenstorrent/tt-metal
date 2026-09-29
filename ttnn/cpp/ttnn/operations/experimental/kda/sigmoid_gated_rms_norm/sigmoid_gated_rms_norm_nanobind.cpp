// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_nanobind.hpp"
#include "sigmoid_gated_rms_norm.hpp"

#include <string>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail {

void bind_sigmoid_gated_rms_norm(nb::module_& mod) {
    ttnn::bind_function<"sigmoid_gated_rms_norm", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply per-head RMS normalization followed by sigmoid (or SiLU) gating.

        For input head ``h``:

            normalized = input / sqrt(mean(input², dim=V) + epsilon)
            z = gate[..., 32*gate_col_offset_tiles + h*V : 32*gate_col_offset_tiles + (h+1)*V]
            output = normalized * weight * sigmoid(z)        # gate_activation="sigmoid"
            output = normalized * weight * z * sigmoid(z)    # gate_activation="silu"

        The operation converts head-first input ``[B*H, T, V]`` into time-first
        output ``[B, T, H*V]`` for the following output projection.

        Args:
            input (ttnn.Tensor): Input tensor ``[B*H, T, V]``. Must be an
                interleaved TILE-layout device tensor with FLOAT32 or BFLOAT16 dtype.
            gate (ttnn.Tensor): Gate ``[B, T, W]`` with
                ``W >= 32*gate_col_offset_tiles + H*V``. Must be an interleaved
                (DRAM or L1) TILE-layout BFLOAT16 device tensor. The op reads the
                ``H*V`` columns that start at tile column ``gate_col_offset_tiles``;
                the tile-row stride is the gate's padded width, so ``gate`` can be a
                wider projection output (for example z|a|b) read in place.
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
            gate_activation (str): ``"sigmoid"`` (default) or ``"silu"``. ``"silu"``
                uses the SFPU SiLU, which is the same non-approximate SFPU sigmoid
                as ``"sigmoid"`` followed by one multiply by the gate value.
            gate_col_offset_tiles (int): First gate tile column to read. Defaults to 0.
            kernel_variant (int, optional): Compute-kernel variant. ``0`` is the
                legacy seven-pass kernel. ``1``-``4`` are the fused kernel, which runs
                one head x 32-row unit in three kinds of DEST pass (sum of squares,
                inverse RMS, output) with a software pipeline across units: ``1``
                library SFPU sigmoid/silu and multiply on the MATH thread; ``2`` one
                fused SFPU pass for the gate (same result as ``1``); ``3`` the SFPU
                work on the PACK thread (same result as ``1``); ``4`` = ``3`` with an
                ``exp_21f`` exponential in the sigmoid (not bit-exact with ``1``-``3``,
                within 1 bf16 ulp of ``0`` for > 99.9% of values). Defaults to
                ``None`` = ``4``.

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
        nb::arg("output_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("gate_activation") = std::string("sigmoid"),
        nb::arg("gate_col_offset_tiles") = 0u,
        nb::arg("kernel_variant") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail
