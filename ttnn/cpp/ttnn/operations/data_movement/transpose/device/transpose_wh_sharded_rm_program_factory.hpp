// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct TransposeWHShardedRMProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const TransposeParams& operation_attributes,
        const TransposeInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::prim
