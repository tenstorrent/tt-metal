// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

struct ExpRingJointSDPASharedVariables {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};  // non-fabric: columns 0..grid_size.x-3
    tt::tt_metal::KernelHandle
        writer_fabric_kernels_id{};  // fabric MUX clients: columns grid_size.x-2 and grid_size.x-1
    tt::tt_metal::KernelHandle compute_kernels_id{};
    // Offset into fabric writer RT args where all-gather args begin (all link_in_range MUX writers)
    uint32_t writer_fabric_ag_rt_offset = 0;
    // Offset into reader RT args where the fused-op global semaphore address lives
    uint32_t reader_fused_op_sem_rt_offset = 0;
    // Offset into reader RT args where per-link semaphore addresses begin
    uint32_t reader_per_link_sem_rt_offset = 0;
    uint32_t num_links = 0;
    // MUX kernel handle and core positions
    tt::tt_metal::KernelHandle ccl_mux_kernel_id{};
    std::vector<CoreCoord> ccl_mux_backward_cores;
    std::vector<CoreCoord> ccl_mux_forward_cores;
};

struct ExpRingJointSDPAProgramFactory {
    using shared_variables_t = ExpRingJointSDPASharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ExpRingJointSDPAParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ExpRingJointSDPAParams& args,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const ExpRingJointSDPAParams& args,
        const ttnn::MeshCoordinate& coord,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& output_tensors);
};

}  // namespace ttnn::prim
