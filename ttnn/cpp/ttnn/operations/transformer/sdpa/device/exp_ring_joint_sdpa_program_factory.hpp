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
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
    // CCL (all-gather) kernel handles
    tt::tt_metal::KernelHandle ccl_reader_forward_kernel_id{};
    tt::tt_metal::KernelHandle ccl_writer_forward_kernel_id{};
    tt::tt_metal::KernelHandle ccl_reader_backward_kernel_id{};
    tt::tt_metal::KernelHandle ccl_writer_backward_kernel_id{};
    std::vector<CoreCoord> ccl_worker_cores;
    uint32_t ccl_num_inputs = 0;
    uint32_t ccl_reader_sender_rt_offset = 0;
    uint32_t ccl_writer_sender_rt_offset = 0;
    uint32_t ccl_num_links = 0;
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
