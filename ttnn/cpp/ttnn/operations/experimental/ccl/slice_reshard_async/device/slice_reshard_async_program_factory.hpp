// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct SliceReshardAsyncProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const SliceReshardAsyncParams& args,
        const Tensor& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied final/barrier GlobalSemaphores and the input/output tensors on every cache hit.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const SliceReshardAsyncParams& args,
        const Tensor& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
