// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/dispatch_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Allocates and returns {dispatched_buffer, metadata}, both per-device and ROW_MAJOR. Same job as
// `dispatch` over a different transport. See the nanobind docstring for what each tensor carries.
std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::dispatch_fabric2d::dispatch_fabric2d;
}  // namespace ttnn
