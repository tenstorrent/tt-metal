// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>

#include <optional>

namespace ttnn::prim {

struct TransposeHCShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const TransposeParams& operation_attributes,
        const TransposeInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
