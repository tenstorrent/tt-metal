// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Same call as ttnn::experimental::deepseek_prefill::combine, plus `expert_offsets`. Parameter names match
// the production op exactly so one caller can dispatch to either without special-casing the argument list;
// `use_fp8_combine` is accepted even though this op rejects it, for that reason.
ttnn::Tensor combine_fabric2d(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis = 0,
    uint32_t num_links = 2,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    bool init_zeros = false,
    bool use_fp8_combine = false);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::combine_fabric2d::combine_fabric2d;
}  // namespace ttnn
