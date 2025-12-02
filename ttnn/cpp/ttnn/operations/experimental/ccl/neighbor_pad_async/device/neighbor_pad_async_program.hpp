// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

#include <optional>

namespace ttnn {

tt::tt_metal::operation::ProgramWithCallbacks neighbor_pad_async_minimal(
    const Tensor& input_tensor,
    const MeshCoordinate& coord,
    std::optional<MeshCoordinate> forward_coord,
    std::optional<MeshCoordinate> backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    ccl::Topology topology,
    uint32_t ring_size,
    uint32_t ring_index,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape);
}  // namespace ttnn
