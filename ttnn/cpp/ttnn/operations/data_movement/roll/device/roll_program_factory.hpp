// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "roll_device_operation_types.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Native single-dim roll for ROW_MAJOR sharded tensors (HEIGHT / WIDTH / BLOCK).
// Implemented as a per-core gather of segment copies over the sharded buffers.
struct RollShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RollParams& operation_attributes, const RollInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RollParams& operation_attributes,
        const RollInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
