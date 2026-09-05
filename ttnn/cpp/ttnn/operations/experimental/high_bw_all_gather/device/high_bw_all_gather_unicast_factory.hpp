// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "high_bw_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <array>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::high_bw_all_gather {

struct HighBwAllGatherUnicastFactory {
    struct shared_variables_t {
        // Slice-order destinations and direction receive counts are fixed at program creation.
        std::vector<std::array<tt::tt_metal::CoreCoord, 2>> destinations;
        std::array<uint32_t, 2> receive_counts{};
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        // Kernel identity is stable; runtime-argument backing storage is resolved afresh per call.
        tt::tt_metal::KernelRuntimeArgsAccessor reader_runtime_args;
        tt::tt_metal::KernelRuntimeArgsAccessor writer_runtime_args;
        tt::tt_metal::GlobalSemaphore ready_sem;
        tt::tt_metal::GlobalSemaphore data_valid_sem;
        uint32_t num_links{};
        uint32_t workers_per_direction{};
        uint32_t num_devices{};
        uint32_t device_idx{};
        uint32_t forward_iterations{};
        uint32_t backward_iterations{};
        uint32_t num_dram_banks{};
        bool is_ring{};
        bool ring_even_split{};
        bool output_bank_owned_schedule{};
    };

    struct workload_state_t {
        // Accessors retain kernel identity across workload moves; never cache runtime payload pointers.
        std::vector<shared_variables_t> programs;
        // Programs in a group have identical slice geometry determinants, but retain their own
        // destination cores, receive counts, accessors, and semaphore owners.
        std::vector<std::vector<size_t>> control_groups;
    };

    using cached_mesh_workload_t = ttnn::device_operation::CachedMeshWorkload<workload_state_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const HighBwAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const HighBwAllGatherParams& operation_attributes,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const HighBwAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const HighBwAllGatherInputs& tensor_args,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& ready_sem,
        const tt::tt_metal::GlobalSemaphore& data_valid_sem);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather
