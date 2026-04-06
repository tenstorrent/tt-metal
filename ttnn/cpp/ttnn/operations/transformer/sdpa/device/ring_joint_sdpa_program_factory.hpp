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
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

namespace ttnn::prim {

struct RingJointSDPASharedVariables {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
    experimental::prim::RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::shared_variables_t
        all_gather_shared_variables;
};

struct RingJointSDPAProgramFactory {
    using shared_variables_t = RingJointSDPASharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RingJointSDPAParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingJointSDPAParams& args,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const RingJointSDPAParams& args,
        const ttnn::MeshCoordinate& coord,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors);
};

}  // namespace ttnn::prim
