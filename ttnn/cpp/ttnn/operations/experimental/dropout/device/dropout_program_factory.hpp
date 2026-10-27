// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "dropout_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {
struct DropoutProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const DropoutParams& args, const DropoutInputs& tensor_args, Tensor& output);

    // Re-derives ALL per-dispatch state (seed, src/dst tensor bindings, per-core tile counts/offsets)
    // on every cache hit -- applied to the cached program in place, no spec rebuild. seed is
    // hash-excluded (per-device offset applied when use_per_device_seed); supersedes the framework's
    // tensor-binding refresh, so both tensor bindings are re-supplied here.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct DropoutMeshWorkloadFactory {
    // Dropout generates N different programs, but they differ only in the per-device seed set as a runtime argument.
    // TODO: when heterogeneous runtime arguments are supported, create a single program for all devices, and only
    // override the runtime arguments for each device. In addition, use `CachedMeshWorkload` instead of
    // `AdaptedCachedMeshWorkload`, as only a single `shared_variables_t` is needed.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const DropoutParams& args,
        const DropoutInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Delegates to DropoutProgramFactory: both factories emit the same per-core layout, so one
    // implementation keeps the cache-hit patch from drifting between them. Called once per range;
    // create_mesh_workload_artifacts emits one single-coordinate range per device, so the range's
    // start coordinate is the device whose seed is re-derived.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
