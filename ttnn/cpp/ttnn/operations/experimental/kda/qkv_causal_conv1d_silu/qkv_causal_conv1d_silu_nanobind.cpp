// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "qkv_causal_conv1d_silu_nanobind.hpp"
#include "pack_convolution_carry.hpp"
#include "qkv_causal_conv1d_silu.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail {
void bind_qkv_causal_conv1d_silu(nb::module_& mod) {
    nb::class_<ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig>(mod, "QkvCausalConv1dSiluProgramConfig")
        .def(nb::init<uint32_t>(), nb::kw_only(), nb::arg("channel_chunk_size").noconvert())
        .def_ro("channel_chunk_size", &ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig::channel_chunk_size)
        .def("__repr__", [](const ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig& config) {
            return fmt::format("QkvCausalConv1dSiluProgramConfig(channel_chunk_size={})", config.channel_chunk_size);
        });

    ttnn::bind_function<"qkv_causal_conv1d_silu", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply a four-tap depthwise causal convolution with SiLU and split the
        result directly into Q, K, and V tensors.

        Let ``x[-3:0]`` be the supplied history and ``x[0:T]`` the current input.
        For each token and channel:

            convolved[t] =
                tap0 * x[t-3] + tap1 * x[t-2] + tap2 * x[t-1] + tap3 * x[t]
            q, k, v = split(silu(convolved), [q_width, k_width, v_width])

        Args:
            input (ttnn.Tensor): Current tokens ``[1, T, Q+K+V]``. Must be an
                interleaved ROW_MAJOR BFLOAT16 device tensor.
            history (ttnn.Tensor): Interleaved ROW_MAJOR BFLOAT16 history.
                With wrap_row=0 its shape is ``[1,3,Q+K+V]``. A nonzero wrap_row
                requires ``[1,6,Q+K+V]`` even when this device's indicator is zero.
                Rows 0:3 seed the physical head; rows 3:6 seed the physical tail
                at an enabled wrap. The caller retains a three-row stream carry.
            tap0, tap1, tap2, tap3 (ttnn.Tensor): Per-channel convolution taps.
                Each must have logical volume ``Q+K+V`` and be an interleaved
                TILE-layout BFLOAT16 device tensor.
            q_width (int): Output Q width.
            k_width (int): Output K width.
            v_width (int): Output V width.

        Keyword Args:
            program_config (QkvCausalConv1dSiluProgramConfig): Required program tuning;
                ``channel_chunk_size`` is expressed in logical channels.
            wrap_row (int, optional): Tile-aligned row strictly inside T, or
                zero to disable wrapping. Defaults to zero.
            wrap_indicator (ttnn.Tensor, optional): Interleaved TILE FLOAT32
                device tensor whose first scalar controls the local wrap.
                Nonzero enables wrap_row; zero ignores the second history plane.
                Without an indicator, a nonzero wrap_row applies unconditionally.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: New TILE-layout BFLOAT16
                tensors ``q[1,T,Q]``, ``k[1,T,K]``, and ``v[1,T,V]``.

        Note:
            ``T``, ``Q``, ``K``, and ``V`` must be positive and tile-aligned.
            All inputs must be allocated on the same device. Inputs, including
            ``history``, are not modified; the caller owns history updates.
        )doc",
        &ttnn::experimental::kda::qkv_causal_conv1d_silu,
        nb::arg("input").noconvert(),
        nb::arg("history").noconvert(),
        nb::arg("tap0").noconvert(),
        nb::arg("tap1").noconvert(),
        nb::arg("tap2").noconvert(),
        nb::arg("tap3").noconvert(),
        nb::arg("q_width"),
        nb::arg("k_width"),
        nb::arg("v_width"),
        nb::kw_only(),
        nb::arg("program_config").noconvert(),
        nb::arg("wrap_row") = 0,
        nb::arg("wrap_indicator") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"pack_convolution_carry", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Pack the one-tile sequence-parallel causal-convolution publication.

        Ordinary devices publish their physical final history. The device whose
        local wrap indicator is nonzero publishes the history immediately before
        wrap_row and retains its physical final history in the next rows.

        Args:
            input (ttnn.Tensor): Interleaved ROW_MAJOR BFLOAT16 ``[1,T,C]``.
                T and C must be positive and tile-aligned.
            wrap_indicator (ttnn.Tensor): Interleaved TILE FLOAT32 tensor on
                the same device; its first scalar is the local predicate.
            wrap_row (int): Tile-aligned boundary in ``[history_rows,T)``.

        Keyword Args:
            history_rows (int): Positive history length H with ``2*H <= 32``;
                defaults to three.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output
                placement; defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Device compute configuration.

        Returns:
            ttnn.Tensor: New BFLOAT16 TILE ``[1,32,C]`` publication. Rows 0:H
                contain input[wrap_row-H:wrap_row] on the boundary device and
                input[T-H:T] elsewhere. Rows H:2H contain input[T-H:T] only on
                the boundary device. All remaining rows are zero. Inputs are
                immutable and the output owns separate storage.
        )doc",
        &ttnn::experimental::kda::pack_convolution_carry,
        nb::arg("input").noconvert(),
        nb::arg("wrap_indicator").noconvert(),
        nb::arg("wrap_row"),
        nb::kw_only(),
        nb::arg("history_rows") = 3,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail
