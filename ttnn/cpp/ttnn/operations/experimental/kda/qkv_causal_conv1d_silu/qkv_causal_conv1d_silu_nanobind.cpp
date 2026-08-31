// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "qkv_causal_conv1d_silu_nanobind.hpp"
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

        Let ``x[-3:-1]`` be the supplied history and ``x[0:T]`` the current input.
        For each token and channel:

            convolved[t] =
                tap0 * x[t-3] + tap1 * x[t-2] + tap2 * x[t-1] + tap3 * x[t]
            q, k, v = split(silu(convolved), [q_width, k_width, v_width])

        Args:
            input (ttnn.Tensor): Current tokens ``[1, T, Q+K+V]``. Must be an
                interleaved ROW_MAJOR BFLOAT16 device tensor.
            history (ttnn.Tensor): The three tokens preceding ``input``, shaped
                ``[1, 3, Q+K+V]``. Must be an interleaved ROW_MAJOR BFLOAT16
                device tensor.
            tap0, tap1, tap2, tap3 (ttnn.Tensor): Per-channel convolution taps.
                Each must have logical volume ``Q+K+V`` and be an interleaved
                TILE-layout BFLOAT16 device tensor.
            q_width (int): Output Q width.
            k_width (int): Output K width.
            v_width (int): Output V width.

        Keyword Args:
            program_config (QkvCausalConv1dSiluProgramConfig): Required program tuning;
                ``channel_chunk_size`` is expressed in logical channels.
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
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail
