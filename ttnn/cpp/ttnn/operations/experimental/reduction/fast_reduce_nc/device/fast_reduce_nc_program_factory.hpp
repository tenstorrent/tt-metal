// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"  // ttnn::MeshCoordinate in override_runtime_arguments()
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "fast_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FastReduceNCProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const FastReduceNCParams& operation_attributes,
        const FastReduceNCInputs& tensor_args,
        std::vector<Tensor>& outputs);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const FastReduceNCParams& operation_attributes,
        const FastReduceNCInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
