// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceTileTensorArgsProgramFactory {
    // The writer binds the Metal 2.0 fork of eltwise/unary's shared interleaved writer, so its DFB
    // accessor name, tensor accessor name and runtime-argument names come from that fork.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: the four tensor bindings plus the per-core scalars, which are hash-excluded.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
