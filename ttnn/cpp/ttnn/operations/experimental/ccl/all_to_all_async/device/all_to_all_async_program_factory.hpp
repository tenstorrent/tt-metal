// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/workload/run_args/program_run_args.hpp>

namespace ttnn::experimental::prim {

struct AllToAllAsyncProgram {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const AllToAllAsyncParams& operation_attributes,
        const AllToAllAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied GlobalSemaphore and the input, intermediate and output tensors on every cache hit.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const AllToAllAsyncParams& operation_attributes,
        const AllToAllAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
