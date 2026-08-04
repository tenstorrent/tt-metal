// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neighbor_pad_halo_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

// Kernel handles + core set for the NP fabric kernels the halo-only program factory builds.
// Retained in the cached program so override_runtime_arguments can refresh per-dispatch args.
struct NpHaloArtifacts {
    tt::tt_metal::KernelHandle h_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle h_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle w_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle w_writer_kernel_id = 0;
    bool has_w_fabric = false;
    tt::tt_metal::CoreRangeSet fabric_core_range;  // all NP fabric cores (H + W)
    // Padded-output fused mode: the concurrent interior-copy scatter kernel (0 if not padded mode).
    bool has_scatter = false;
    tt::tt_metal::KernelHandle scatter_kernel_id = 0;
    tt::tt_metal::CoreRangeSet scatter_core_range;
};

struct NpHaloSharedVariables {
    NpHaloArtifacts np_artifacts;
};

struct NpHaloMeshWorkloadFactory {
    using shared_variables_t = NpHaloSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const NpHaloParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const NpHaloInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const NpHaloParams& operation_attributes,
        const NpHaloInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const NpHaloParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const NpHaloInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
