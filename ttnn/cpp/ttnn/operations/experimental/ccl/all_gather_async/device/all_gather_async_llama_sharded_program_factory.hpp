// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/workload/run_args/program_run_args.hpp>

namespace ttnn::experimental::prim {

struct LlamaShardedMeshWorkloadFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied GlobalSemaphores and the input and output tensors on every cache hit.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
