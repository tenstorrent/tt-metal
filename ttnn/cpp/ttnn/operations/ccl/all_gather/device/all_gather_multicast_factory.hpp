// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::ccl {

// Query the same bounded host resource/mapping proof and calibrated automatic
// policy used by the multicast factory. Program-factory selection uses this to
// keep winning ring cases on multicast without bypassing the established
// unicast path for cases where receiver staging regresses.
bool should_auto_select_receiver_l1_path(
    const AllGatherParams& operation_attributes, const AllGatherInputs& tensor_args, const Tensor& output_tensor);

struct AllGatherMulticastFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> worker_cores;
        std::vector<tt::tt_metal::CoreCoord> receiver_cores;
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle receiver_kernel_id{};
        tt::tt_metal::KernelHandle receiver_reader_kernel_id{};
        uint32_t receiver_drain_risc_count = 0;
        bool bank_owned_links = false;
        tt::tt_metal::GlobalSemaphore barrier_sem;
        // Receiver path only: [0] is the sender credit counter, [1] is the
        // local consumed sequence, and [2 + source] is that source's produced
        // sequence on every mirrored receiver core.  When dual-RISC drain is
        // selected, [2 + num_devices] is its local window-completion counter.
        std::vector<tt::tt_metal::GlobalSemaphore> receiver_control_sems;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherParams& operation_attributes,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const AllGatherInputs& tensor_args,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& barrier_sem,
        const std::vector<tt::tt_metal::GlobalSemaphore>& receiver_control_sems);
};

}  // namespace ttnn::operations::ccl
