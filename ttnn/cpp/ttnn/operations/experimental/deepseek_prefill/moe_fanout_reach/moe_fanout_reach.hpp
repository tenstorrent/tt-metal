// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

// This device's row of the multicast reach table dispatch_fabric2d(fanout=True) sizes its chunks
// from: [1, 2, cluster_axis_extent / 2 + 2] INT32, ROW_MAJOR, per device. See the nanobind docstring.
ttnn::Tensor moe_fanout_reach(
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_dispatch_table,
    const ttnn::Tensor& global_dispatch_offsets,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t dispatch_group_size,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_fanout_reach::moe_fanout_reach;
}  // namespace ttnn
