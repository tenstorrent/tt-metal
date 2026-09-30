// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "qkv_causal_conv1d_silu_nanobind.hpp"

#include <optional>
#include <tuple>
#include <variant>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "qkv_causal_conv1d_silu.hpp"
#include "device/qkv_causal_conv1d_silu_tiled_program_factory.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail {
namespace {

using ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig;

// (q, k, v), or (q, k, v, new_state) with return_conv_state=True.
using QkvCausalConv1dSiluResult = std::variant<
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>,
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>>;

QkvCausalConv1dSiluResult qkv_causal_conv1d_silu_binding(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool return_conv_state,
    const std::optional<ttnn::Tensor>& conv_state_output) {
    TT_FATAL(
        return_conv_state || !conv_state_output.has_value(),
        "qkv_causal_conv1d_silu: conv_state_output needs return_conv_state=True (it is the new_state output)");
    if (return_conv_state) {
        return ttnn::experimental::kda::qkv_causal_conv1d_silu_with_conv_state(
            input,
            history,
            tap0,
            tap1,
            tap2,
            tap3,
            q_width,
            k_width,
            v_width,
            program_config,
            memory_config,
            compute_kernel_config,
            conv_state_output);
    }
    return ttnn::experimental::kda::qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config,
        memory_config,
        compute_kernel_config);
}

nb::dict tiled_program_plan_binding(
    uint32_t sequence,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t grid_x,
    uint32_t grid_y,
    const std::optional<uint32_t>& channel_chunk_size,
    bool has_history,
    bool return_conv_state,
    uint32_t tile_size,
    bool conv_state_inplace) {
    namespace prim = ttnn::experimental::prim;
    const uint32_t chunk = channel_chunk_size.value_or(
        prim::qkv_causal_conv1d_silu_tiled::default_channel_chunk_size(q_width, k_width, v_width));
    const auto plan = prim::make_qkv_causal_conv1d_silu_tiled_plan(
        tt::tt_metal::CoreCoord{grid_x, grid_y},
        sequence,
        q_width,
        k_width,
        v_width,
        chunk,
        has_history,
        return_conv_state,
        tile_size,
        conv_state_inplace);

    nb::list dataflow_buffers;
    for (const auto& buffer : plan.dataflow_buffers) {
        nb::dict entry;
        entry["name"] = buffer.name;
        entry["producer"] = buffer.producer;
        entry["consumer"] = buffer.consumer;
        entry["num_entries"] = buffer.num_entries;
        entry["entry_size"] = buffer.entry_size;
        entry["bytes"] = buffer.bytes();
        dataflow_buffers.append(entry);
    }
    nb::dict scratchpad;
    scratchpad["bytes"] = plan.scratch_bytes;
    scratchpad["align_slack"] = plan.scratch_align_slack;
    scratchpad["halo_offset"] = plan.scratch_halo_offset;
    scratchpad["halo_bytes"] = plan.scratch_halo_bytes;
    scratchpad["zeros_offset"] = plan.scratch_zeros_offset;
    scratchpad["zeros_bytes"] = plan.scratch_zeros_bytes;
    scratchpad["state_offset"] = plan.scratch_state_offset;
    scratchpad["state_bytes"] = plan.scratch_state_bytes;
    scratchpad["stage_offset"] = plan.scratch_stage_offset;
    scratchpad["stage_bytes"] = plan.scratch_stage_bytes;

    nb::list cores;
    nb::list step_start;
    nb::list step_count;
    for (size_t i = 0; i < plan.work.cores.size(); ++i) {
        cores.append(nb::make_tuple(plan.work.cores[i].x, plan.work.cores[i].y));
        step_start.append(plan.work.wi_start[i]);
        step_count.append(plan.work.wi_count[i]);
    }

    nb::dict result;
    result["sequence"] = plan.sequence;
    result["widths"] = nb::make_tuple(plan.q_width, plan.k_width, plan.v_width);
    result["channel_chunk_size"] = plan.channel_chunk_size;
    result["block_tiles"] = plan.block_tiles;
    result["Mt"] = plan.Mt;
    result["Qt"] = plan.Qt;
    result["Kt"] = plan.Kt;
    result["Vt"] = plan.Vt;
    result["Ct"] = plan.Ct;
    result["num_blocks"] = plan.num_blocks;
    result["num_steps"] = plan.num_steps;
    result["has_history"] = plan.has_history;
    result["return_conv_state"] = plan.return_conv_state;
    result["conv_state_inplace"] = plan.conv_state_inplace;
    result["tile_size"] = plan.tile_size;
    result["dataflow_buffers"] = dataflow_buffers;
    result["scratchpad"] = scratchpad;
    result["dfb_bytes_per_core"] = plan.dfb_bytes_per_core;
    result["l1_bytes_per_core"] = plan.l1_bytes_per_core;
    result["grid"] = nb::make_tuple(plan.grid.x, plan.grid.y);
    result["num_cores"] = plan.num_cores();
    result["cores"] = cores;
    result["step_start"] = step_start;
    result["step_count"] = step_count;
    result["min_steps_per_core"] = plan.min_steps_per_core;
    result["max_steps_per_core"] = plan.max_steps_per_core;
    result["max_tap_loads_per_core"] = plan.max_tap_loads_per_core;
    result["balance"] = plan.balance;
    result["summary"] = plan.to_string();
    return result;
}

}  // namespace

void bind_qkv_causal_conv1d_silu(nb::module_& mod) {
    nb::class_<QkvCausalConv1dSiluProgramConfig>(mod, "QkvCausalConv1dSiluProgramConfig")
        .def(
            "__init__",
            [](QkvCausalConv1dSiluProgramConfig* self, uint32_t channel_chunk_size, bool fused_qk_l2_norm) {
                new (self) QkvCausalConv1dSiluProgramConfig{channel_chunk_size, fused_qk_l2_norm};
            },
            nb::kw_only(),
            nb::arg("channel_chunk_size").noconvert(),
            nb::arg("fused_qk_l2_norm").noconvert() = false)
        .def_ro("channel_chunk_size", &QkvCausalConv1dSiluProgramConfig::channel_chunk_size)
        .def_ro("fused_qk_l2_norm", &QkvCausalConv1dSiluProgramConfig::fused_qk_l2_norm)
        .def("__repr__", [](const QkvCausalConv1dSiluProgramConfig& config) {
            return fmt::format(
                "QkvCausalConv1dSiluProgramConfig(channel_chunk_size={}, fused_qk_l2_norm={})",
                config.channel_chunk_size,
                config.fused_qk_l2_norm);
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

        The layout of ``input`` selects the program:

        - ROW_MAJOR ``input``: ``history`` and ``program_config`` are required,
          and ``return_conv_state`` must be False.
        - TILE ``input`` (tiled path): ``input`` can come directly from a TILE
          matmul output. ``history`` is a TILE tensor or None. The block size is
          ``B = channel_chunk_size / 32``. For the same bf16 values, q/k/v are
          bit-identical to the ROW_MAJOR path.

        Args:
            input (ttnn.Tensor): Current tokens ``[1, T, Q+K+V]``. Must be an
                interleaved BFLOAT16 device tensor, ROW_MAJOR or TILE (32x32
                tiles).
            history (ttnn.Tensor or None): The three tokens preceding ``input``,
                shaped ``[1, 3, Q+K+V]``. Must be an interleaved BFLOAT16 device
                tensor with the same layout as ``input``. Required for ROW_MAJOR
                input. For TILE input, None means three zero rows.
            tap0, tap1, tap2, tap3 (ttnn.Tensor): Per-channel convolution taps.
                Each must have logical volume ``Q+K+V`` and be an interleaved
                TILE-layout BFLOAT16 device tensor.
            q_width (int): Output Q width.
            k_width (int): Output K width.
            v_width (int): Output V width.

        Keyword Args:
            program_config (QkvCausalConv1dSiluProgramConfig, optional): Program
                tuning; ``channel_chunk_size`` is expressed in logical channels.
                Required for ROW_MAJOR input. For TILE input it sets
                ``B = channel_chunk_size / 32``, with B in {1, 2, 4, 8} and B a
                divisor of ``(Q+K+V) / 32``. For TILE input the default is B = 4
                (``channel_chunk_size=128``), or 2 or 1 when 4 does not divide
                ``(Q+K+V) / 32``.
                ``fused_qk_l2_norm=True`` (TILE input only; needs
                ``channel_chunk_size=128`` and Q, K multiples of 128) selects the
                fast kernel (taps accumulated in dest, TTI SiLU; not bit-identical
                to the default) with a fused per-128-channel-head L2 norm of q and
                k: ``q = y_q * rsqrt(sum(y_q^2) + 1e-6) / sqrt(128)``,
                ``k = y_k * rsqrt(sum(y_k^2) + 1e-6)``. q and k are then FLOAT32;
                v and ``new_state`` stay BFLOAT16. Defaults to False.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration for q, k and v. Defaults to DRAM. ``new_state`` is
                always DRAM interleaved.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.
            return_conv_state (bool, optional): TILE input only. When True, the op
                also returns ``new_state``. Defaults to False.
            conv_state_output (ttnn.Tensor, optional): With ``return_conv_state=True``
                only: a pre-allocated ``new_state`` (interleaved TILE BFLOAT16
                ``[1, 3, Q+K+V]``, DRAM or L1). The op writes ``new_state`` into it,
                allocates none, and returns it. It may be ``history`` itself: an
                in-place conv-state update (for a persistent traced state buffer).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: New TILE-layout BFLOAT16
                tensors ``q[1,T,Q]``, ``k[1,T,K]``, and ``v[1,T,V]`` (q and k are
                FLOAT32 with ``fused_qk_l2_norm=True``).
            With ``return_conv_state=True``: ``(q, k, v, new_state)``, where
                ``new_state`` is a new DRAM-interleaved TILE BFLOAT16 tensor
                ``[1, 3, Q+K+V]`` that holds ``x[T-3:T]`` (the next call's
                ``history``); its tile padding rows are zero.

        Note:
            ``T``, ``Q``, ``K``, and ``V`` must be positive and tile-aligned.
            All inputs must be allocated on the same device. Inputs are not
            modified, except ``history`` when it is also ``conv_state_output``;
            ``new_state`` is a new tensor unless ``conv_state_output`` is given.
        )doc",
        &qkv_causal_conv1d_silu_binding,
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
        nb::arg("program_config").noconvert() = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("return_conv_state") = false,
        nb::arg("conv_state_output") = nb::none());

    mod.def(
        "qkv_causal_conv1d_silu_tiled_program_plan",
        &tiled_program_plan_binding,
        R"doc(
        Describe the program of the TILE-input path of ``qkv_causal_conv1d_silu``
        without a device: the per-core DFB table, the reader scratchpad layout,
        and the block-major work split. The tiled program factory uses the same
        plan, so the numbers here are the numbers it uses.

        Args:
            sequence (int): T, a positive multiple of 32.
            q_width, k_width, v_width (int): Output widths, multiples of 32.

        Keyword Args:
            grid_x, grid_y (int): Compute grid, for example
                ``device.compute_with_storage_grid_size()`` (13 x 10 on P150).
            channel_chunk_size (int, optional): 32 * B. Defaults to the op default.
            has_history (bool, optional): Whether a history tensor is bound.
                Defaults to True.
            return_conv_state (bool, optional): Whether new_state is written.
                Defaults to False.
            tile_size (int, optional): Bytes per tile. Defaults to 2048 (BFLOAT16).
            conv_state_inplace (bool, optional): Whether new_state is written in
                place into history (adds the reader's stage region). Defaults to False.

        Returns:
            dict: Plan fields (``block_tiles``, ``dataflow_buffers``,
                ``scratchpad``, ``dfb_bytes_per_core``, ``l1_bytes_per_core``,
                ``num_cores``, ``step_start``, ``step_count``, ``balance``,
                ...) and a text ``summary``.
        )doc",
        nb::arg("sequence"),
        nb::arg("q_width"),
        nb::arg("k_width"),
        nb::arg("v_width"),
        nb::kw_only(),
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("channel_chunk_size") = nb::none(),
        nb::arg("has_history") = true,
        nb::arg("return_conv_state") = false,
        nb::arg("tile_size") = 2048u,
        nb::arg("conv_state_inplace") = false);
}
}  // namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail
