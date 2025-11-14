// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

/**
 * Fused Broadcast Program Factory
 *
 * Creates a program that implements the fused TP P2P replicate + SP broadcast operation.
 * This is designed for the MLA+MoE latency optimization described in the specification.
 */
tt::tt_metal::operation::ProgramWithCallbacks fused_broadcast_multicore(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const MeshCoordinate& device_coord,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    tt::tt_fabric::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

}  // namespace ttnn
