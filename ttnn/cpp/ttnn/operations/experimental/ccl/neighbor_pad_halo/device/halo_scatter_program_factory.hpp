// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "halo_scatter_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

struct NpHaloScatterArtifacts {
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::CoreCoord core{0, 0};
};

struct NpHaloScatterSharedVariables {
    NpHaloScatterArtifacts artifacts;
};

struct NpHaloScatterMeshWorkloadFactory {
    using shared_variables_t = NpHaloScatterSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const NpHaloScatterParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const NpHaloScatterInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const NpHaloScatterParams& operation_attributes,
        const NpHaloScatterInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const NpHaloScatterParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const NpHaloScatterInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
