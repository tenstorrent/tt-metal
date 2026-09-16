// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Retile factory: accepts an already-tiled input whose tile shape differs from the
// tile shape requested on the op, and re-lays it out into the requested tile shape.
struct TilizeMultiCoreRetileProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const TilizeParams& operation_attributes,
        const TilizeInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};
}  // namespace ttnn::prim
