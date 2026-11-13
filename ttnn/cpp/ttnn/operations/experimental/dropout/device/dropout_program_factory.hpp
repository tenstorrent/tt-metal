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

    // Cache hit: fresh src/dst tensor bindings and the call's seed (hash-excluded). The per-core args derive from the
    // input spec and the compute grid, fixed for a cache entry, so they keep their cache-miss values.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// use_per_device_seed: every device draws its own mask. The devices' programs differ only in the compute kernels'
// common `device_seed_offset`, so each range runs one program with the offset set per device (per-device run args, see
// MeshWorkloadSpecFactoryAdapter).
struct DropoutMeshWorkloadFactory {
    static constexpr bool uses_per_device_run_args = true;

    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const DropoutParams& args,
        const DropoutInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Called once per range: the single-device hit args, plus device_seed_offset = 0 in the base and each device's
    // offset.
    static ttnn::device_operation::RangeRunArgs override_runtime_arguments(
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
