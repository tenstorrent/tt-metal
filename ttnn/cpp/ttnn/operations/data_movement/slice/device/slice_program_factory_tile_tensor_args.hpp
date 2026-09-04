// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <optional>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct SliceTileTensorArgsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
