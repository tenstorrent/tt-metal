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

        Keyword Args:
            reduce_dim (int): dimension along which to reduce (typically 0 for experts_k).
            split_size (int): size of last dim of each output split tensor
                (typically hidden_size // num_replicated_devices).
            output_memory_config (ttnn.MemoryConfig): output memory configuration.
                Supports both interleaved and sharded (NdShardSpec) layouts.
            scores_tensor (ttnn.Tensor, optional): expert scores [tokens, 1, seq, experts_k],
                ROW_MAJOR layout, DRAM. If omitted, calls :func:`~ttnn.experimental.deepseek_moe_fast_reduce_nc`
                instead (no fused multiply by scores).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): optional compute config.

        Returns:
            list of ttnn.Tensor: (hidden_size // split_size) output tensors, each with shape
                [1, 1, tokens, split_size].
        )doc",
        &ttnn::experimental::reduction::deepseek_moe_fast_reduce_nc_fused,
        nb::arg("input_tensor"),
        nb::arg("reduce_dim"),
        nb::kw_only(),
        nb::arg("split_size"),
        nb::arg("output_memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        nb::arg("scores_tensor") = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
