// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/workload/run_args/program_run_args.hpp>

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
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct DropoutMeshWorkloadFactory {
    // The devices' programs differ only in the compute kernels' per-core seeds (each device offsets the seed by its
    // id), so each range runs one program with those seeds as per-device run args.
    static constexpr bool uses_per_device_run_args = true;

    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const DropoutParams& args,
        const DropoutInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Called once per range: DropoutProgramFactory's run args for the range's first device as the base, plus every
    // device's per-core seeds.
    static ttnn::device_operation::RangeRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
