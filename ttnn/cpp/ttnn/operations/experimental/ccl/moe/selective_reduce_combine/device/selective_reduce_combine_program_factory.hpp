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

// Appends CBs and kernels onto desc. nullopt semaphores bake address 0 (local_combine / FullLocal).
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
    uint32_t compute_cores_per_combine_cores = 0,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id = std::nullopt);

}  // namespace ttnn::experimental::prim
