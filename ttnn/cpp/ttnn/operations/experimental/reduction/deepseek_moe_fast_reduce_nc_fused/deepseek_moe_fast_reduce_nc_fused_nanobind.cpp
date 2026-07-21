// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "deepseek_moe_fast_reduce_nc_fused_nanobind.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_deepseek_moe_fast_reduce_nc_fused(nb::module_& mod) {
    ttnn::bind_function<"deepseek_moe_fast_reduce_nc_fused", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused operation combining permute + tilize + mul(activation, expert_scores) + deepseek_moe_fast_reduce_nc
        into a single kernel launch.

        The operation eliminates the large intermediate scaled-activation tensor by applying the
        per-expert score scale inside the reduce loop using a hardware multiply-accumulate (MAC)
        instruction with BroadcastType::COL.

        Args:
            input_tensor (ttnn.Tensor): activation tensor [experts_k, 1, tokens, hidden_size],
                TILE layout, L1.
            expert_indices_tensor (ttnn.Tensor): per-token expert indices
                (matches all_to_all_dispatch convention).
            expert_mapping_tensor (ttnn.Tensor): expert-to-device mapping
                (matches all_to_all_dispatch convention).
            reduce_dim (int): dimension along which to reduce (typically 0 for experts_k).

        Keyword Args:
            split_size (int): size of last dim of each output split tensor
                (typically hidden_size // num_replicated_devices).
            cluster_axis (int): mesh axis (0 or 1) along which the expert mapping is laid out.
            output_memory_config (ttnn.MemoryConfig): output memory configuration.
                Supports both interleaved and sharded (NdShardSpec) layouts.
            scores_tensor (ttnn.Tensor, optional): expert scores [tokens, 1, seq, experts_k],
                ROW_MAJOR layout, DRAM. If omitted, calls :func:`~ttnn.experimental.deepseek_moe_fast_reduce_nc`
                instead (no fused multiply by scores). The expert_indices_tensor /
                expert_mapping_tensor / cluster_axis arguments are ignored on that fallback path.
            num_shared_experts (int, optional): number of shared experts occupying the trailing
                slots of the reduce dimension, whose size is
                ``reduction_dim_size = num_routed_experts + num_shared_experts``. The leading
                ``num_routed_experts`` slots are scaled by per-token ``scores_tensor`` values; the
                trailing shared-expert slots are scaled by the constant ``shared_expert_scale``
                instead. The ``scores_tensor`` last dim must equal ``num_routed_experts``
                (``= reduction_dim_size - num_shared_experts``). This is the *logical* shared-expert
                count — matching the all_to_all_dispatch / ``select_experts_k + num_shared_experts``
                layout — NOT the per-device ``num_shared_experts_per_device`` used by
                ``ttnn.experimental.moe_compute``. Defaults to 0.
            shared_expert_scale (float, optional): constant scale applied to each shared expert's
                contribution inside the reduce loop, used in place of the per-token scores that
                exist only for routed experts. Defaults to 1.0.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): optional compute config.

        Returns:
            list of ttnn.Tensor: (hidden_size // split_size) output tensors, each with shape
                [1, 1, tokens, split_size].
        )doc",
        &ttnn::experimental::reduction::deepseek_moe_fast_reduce_nc_fused,
        nb::arg("input_tensor"),
        nb::arg("expert_indices_tensor"),
        nb::arg("expert_mapping_tensor"),
        nb::arg("reduce_dim"),
        nb::kw_only(),
        nb::arg("split_size"),
        nb::arg("cluster_axis"),
        nb::arg("output_memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        nb::arg("scores_tensor") = nb::none(),
        nb::arg("num_shared_experts") = 0,
        nb::arg("shared_expert_scale") = 1.0f,
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
