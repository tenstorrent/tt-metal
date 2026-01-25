// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

// Include the new device operation types to get ReduceScatterProgramArtifacts
#include "reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operation.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

// Import the ReduceScatterProgramArtifacts from the prim namespace for backward compatibility
using ReduceScatterProgramArtifacts = experimental::prim::ReduceScatterProgramArtifacts;

void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor);

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset);

void line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output);

void ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output);

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset);

}  // namespace ttnn
