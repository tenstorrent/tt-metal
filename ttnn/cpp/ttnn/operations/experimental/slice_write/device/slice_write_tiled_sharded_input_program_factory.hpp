// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "slice_write_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct SliceWriteTiledShardedInputProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value);
};

// The tiled sharded-input program with runtime_shapes: keyed without the start and the output's outer and spatial
// sizes (only its channel count), so it serves every slice of the same extent into outputs of any size. It clips the
// slice to the output's rows and tile columns, so the H and W ends may lie past the output. Every per-core arg is
// recomputed on a hit.
struct SliceWriteTiledShardedInputRuntimeShapesProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceWriteParams& operation_attributes,
        const SliceWriteInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim
