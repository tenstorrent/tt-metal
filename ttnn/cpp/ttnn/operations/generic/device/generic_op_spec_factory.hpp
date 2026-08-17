// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "generic_op_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::generic::program {

// CustomProgramSpecFactoryConcept: create_program_artifacts on the miss, and
// override_runtime_arguments on every hit. The override is mandatory here, not an
// optimisation -- the caller rebuilds run args (including tensor args) per call, and the
// custom-spec adapter's hit path applies ONLY what this returns.
struct GenericSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::generic::program
