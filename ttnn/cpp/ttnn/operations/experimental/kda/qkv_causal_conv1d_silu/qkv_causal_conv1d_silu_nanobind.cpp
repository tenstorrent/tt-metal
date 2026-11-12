// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "qkv_causal_conv1d_silu_nanobind.hpp"
#include "qkv_causal_conv1d_silu.hpp"
#include "device/qkv_causal_conv1d_silu_program_factory.hpp"
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

        At the logical sequence start, preceding tokens come from the caller's
        incoming ``history``. Other ranks begin with ``predecessor_carry``. On a
        rank containing both the chronological head and tail, the tail also
        restarts from ``predecessor_carry`` instead of the adjacent local head
        rows. The kernel derives these boundaries from ``actual_start`` and its
        mesh coordinate.

        Args:
            input (ttnn.Tensor): Interleaved TILE BFLOAT16 ``[1, T, W]`` whose
                channels from ``input_column_offset`` are the convolution input,
                read in place (such as a fused projection). A ROW_MAJOR input is
                tilized first.
            history (ttnn.Tensor): Interleaved ROW_MAJOR BFLOAT16 history.
                Shape ``[1,3,Q+K+V]``. SP calls supply the predecessor's three-row
                history separately. The kernel derives when each history is needed
                from actual_start and its mesh coordinate.
            tap0, tap1, tap2, tap3 (ttnn.Tensor): Per-channel convolution taps.
                Each must have logical volume ``Q+K+V`` and be an interleaved
                TILE-layout BFLOAT16 device tensor.
            q_width (int): Output Q width.
            k_width (int): Output K width.
            v_width (int): Output V width.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar
                containing the absolute position of the chunk's first token; pass [0]
                for zero-offset execution. Its
                value must be nonnegative and 32-aligned. Keep its address stable
                and update its contents before replay of a captured trace.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the
                sequence. Native mesh coordinates supply each device's rank.
            predecessor_carry (ttnn.Tensor): Three-row history from the
                preceding rank, matching history. For local execution, alias history.
            program_config (QkvCausalConv1dSiluProgramConfig): Required program tuning;
                ``channel_chunk_size`` is expressed in logical channels.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.
            input_column_offset (int, optional): First convolution channel of
                ``input``; tile aligned. Defaults to 0.

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
        nb::arg("actual_start").noconvert(),
        nb::arg("predecessor_carry").noconvert(),

        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("input_column_offset") = 0);

    mod.def(
        "qkv_causal_conv1d_silu_exchange_core",
        &ttnn::experimental::prim::qkv_causal_conv1d_silu_exchange_core,
        nb::arg("grid"),
        R"doc(
        The core of a ``grid``-sized compute grid that runs
        ``qkv_causal_conv1d_silu_with_carry``'s fabric history exchange, and so
        the core its ``exchange_semaphores`` must cover.
        )doc");

    ttnn::bind_function<"qkv_causal_conv1d_silu_with_carry", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        ``qkv_causal_conv1d_silu`` that also exchanges the sequence-parallel
        histories itself and returns the replacement carry. Every rank must call
        it.

        ``exchange_buffer`` is a persistent ``[1, 12, Q+K+V]`` ROW_MAJOR BFLOAT16
        DRAM tensor and ``exchange_semaphores`` three persistent global semaphores
        on ``qkv_causal_conv1d_silu_exchange_core``, zero before the first call. Each segment
        that does not start the logical sequence reads the preceding rank's last
        three rows, which arrive over the fabric. The fourth output is the
        three-token history ending at the logical end of the interval
        (``actual_end``, default full capacity), for the next call.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: ``q``, ``k``,
                ``v`` as ``qkv_causal_conv1d_silu``, and the ROW_MAJOR BFLOAT16
                ``[1, 3, Q+K+V]`` replacement carry, replicated across SP.
        )doc",
        &ttnn::experimental::kda::qkv_causal_conv1d_silu_with_carry,
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
        nb::arg("actual_start").noconvert(),
        nb::arg("exchange_buffer").noconvert(),
        nb::arg("exchange_semaphores"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("input_column_offset") = 0);
}
}  // namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail
