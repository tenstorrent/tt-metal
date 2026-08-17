// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "slice_write_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim::qsr {

// Metal-2.0 (Quasar) tiled slice_write, sharded input -> interleaved output. Mirror of the RM
// sharded-input factory (borrowed INPUT DFB; reader marks the resident TILE shard available; the writer
// drains it and writes each TILE to the interleaved output at start_id + the padded-tile-dim walk).
struct SliceWriteTiledShardedInputProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
