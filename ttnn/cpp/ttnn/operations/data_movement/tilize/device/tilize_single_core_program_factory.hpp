// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {
struct TilizeSingleCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const TilizeParams& operation_attributes,
        const TilizeInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};
}  // namespace ttnn::prim
