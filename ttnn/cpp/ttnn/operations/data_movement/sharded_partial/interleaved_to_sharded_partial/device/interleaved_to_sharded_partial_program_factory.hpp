// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "interleaved_to_sharded_partial_op_types.hpp"

namespace ttnn::prim {

struct InterleavedToShardedPartialProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const InterleavedToShardedPartialParams& params, const Tensor& input, Tensor& output);

    // slice_index is excluded from the program hash, so a hit for a different slice must re-derive
    // starting_idx_h and the buffer addresses; see the .cpp for the patched slots.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const InterleavedToShardedPartialParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
