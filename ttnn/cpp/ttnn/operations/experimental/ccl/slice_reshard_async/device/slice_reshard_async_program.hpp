// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

#include <optional>

namespace ttnn {

tt::tt_metal::operation::ProgramWithCallbacks slice_reshard_async_minimal(
    const Tensor& input_tensor,
    tt::tt_fabric::FabricNodeId sender_fabric_node_id,
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id,
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t output_dim_offset,
    uint32_t output_dim_shape,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    ccl::Topology topology,
    uint32_t ring_size,
    uint32_t ring_index);

}  // namespace ttnn
