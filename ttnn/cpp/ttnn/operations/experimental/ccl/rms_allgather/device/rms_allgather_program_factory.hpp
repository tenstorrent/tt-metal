// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rms_allgather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct RMSAllGatherSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    tt::tt_metal::KernelHandle writer_mcast_sender_kernels_id{};
    tt::tt_metal::KernelHandle writer_mcast_receiver_kernels_id{};
    uint32_t num_none_all_to_all_workers = 0;
    tt::tt_metal::CBHandle pre_cb_in0{};
    tt::tt_metal::CBHandle cb_in1{};
    tt::tt_metal::CBHandle cb_add_out{};
    tt::tt_metal::CBHandle cb_in0{};
    tt::tt_metal::CBHandle cb_stats{};
    tt::tt_metal::CBHandle cb_output{};
    tt::tt_metal::CBHandle cb_output_reshard{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct RMSAllGatherMeshWorkloadFactory {
    using shared_variables_t = RMSAllGatherSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RMSAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const RMSAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
