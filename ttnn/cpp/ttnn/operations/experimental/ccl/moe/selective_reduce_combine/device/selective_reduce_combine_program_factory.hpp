// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "selective_reduce_combine_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

namespace detail {

struct SelectiveReduceCombineWorkerLayout {
    std::vector<uint32_t> data_parallel_sizes_bytes;
    uint32_t num_data_parallel_cores = 0;
    uint32_t num_worker_cores = 0;
};

struct FusedSourceBufferLayout {
    uint32_t rows_per_buffer = 0;             // token-segment rows per ring entry
    uint32_t buffer_block_size_bytes = 0;     // rows_per_buffer * token_segment_size_bytes
    uint32_t circular_buffer_size_bytes = 0;  // num_buffers * buffer_block_size_bytes == the shard
};

// Fused source: the producer (moe_compute dm1) and the combine writer alias one
// height-sharded L1 tensor as a num_buffers-entry ring. Each core's shard is
// [source_shard_height, source_shard_width] elements, but both sides address it in
// rows of one token segment (token_segment_width = hidden_size / num_data_parallel_cores
// elements, token_segment_size_bytes each): dm1's token_expert_row_offset and the
// writer's block stride count token-segment rows, not shard rows. A ring entry holds
// source_shard_height / num_buffers shard rows, each source_shard_width /
// token_segment_width segments wide, so
//   rows_per_buffer = source_shard_height / num_buffers * (source_shard_width / token_segment_width)
// (equal to the shard height split when the shard is one segment wide). The shard
// height must divide by num_buffers, the shard width by the segment width, and the
// ring (== the shard's bytes) must fit the L1 bank (source_buffer_size_bytes).
FusedSourceBufferLayout compute_fused_source_buffer_layout(
    uint32_t source_shard_height,
    uint32_t source_shard_width,
    uint32_t token_segment_width,
    uint32_t source_buffer_size_bytes,
    uint32_t token_segment_size_bytes,
    uint32_t num_buffers);

SelectiveReduceCombineWorkerLayout compute_worker_layout(
    const Tensor& input_tensor,
    uint32_t hidden_size,
    uint32_t num_token_parallel_cores,
    uint32_t num_data_parallel_cores,
    bool local_combine = false);

}  // namespace detail

struct UnifiedSelectReduce {
    using operation_attributes_t = SelectiveReduceCombineParams;
    using tensor_args_t = SelectiveReduceCombineTensors;
    using tensor_return_value_t = ttnn::Tensor;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

// Appends CBs and kernels onto desc. nullopt semaphores bake address 0 (local_combine / SingleDevice).
void append_selective_reduce_combine_to_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_semaphore,
    const std::optional<GlobalSemaphore>& cross_device_semaphore,
    uint32_t metadata_sync_semaphore_id,
    uint32_t compute_sync_semaphore_id,
    // Fused moe_compute only: for each data-parallel (width) column, the compute cores whose
    // output feeds that column. Presence switches the writer to a double-buffered source.
    const std::optional<std::vector<std::vector<CoreCoord>>>& compute_cores_by_combine_column = std::nullopt);

}  // namespace ttnn::experimental::prim
