// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neighbor_pad_conv3d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

// Kernel handles + core set for the NP fabric kernels the fused program factory builds
// inline. Retained in the cached program so override_runtime_arguments can refresh their
// per-dispatch runtime args.
struct NpFabricOnlyArtifacts {
    tt::tt_metal::KernelHandle h_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle h_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle w_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle w_writer_kernel_id = 0;
    bool has_w_fabric = false;
    tt::tt_metal::CoreRangeSet fabric_core_range;  // all NP fabric cores (H + W)
};

struct NpConv3dSharedVariables {
    NpFabricOnlyArtifacts np_artifacts;
    // Conv3d kernels
    uint32_t conv3d_num_cores = 0;
    std::vector<CoreCoord> conv3d_cores;
    tt::tt_metal::KernelHandle conv3d_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle conv3d_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle conv3d_compute_kernel_id = 0;
};

struct NpConv3dMeshWorkloadFactory {
    using shared_variables_t = NpConv3dSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const NpConv3dParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const NpConv3dInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const NpConv3dParams& operation_attributes,
        const NpConv3dInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const NpConv3dParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const NpConv3dInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
