// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Sharded retile factory: re-lays an already-tiled, sharded input into the requested tile shape
// (changing tile height) while keeping the data resident in L1.
//
// Each core operates independently on its own shard: the retile compute kernel untilizes the shard
// into a local row-major intermediate and re-tilizes it into the output tile shape. Because the
// element geometry of a shard is unchanged by a retile (only the tiling of those elements changes),
// the output shares the input shard grid and per-shard element shape.
//
// Output paths (selected at runtime by the output memory layout):
//   - Same-layout sharded L1 output: zero-copy (input and output CBs aliased to the shard buffers).
//   - L1 INTERLEAVED output: zero-copy input read, TensorAccessor scatter write. Only valid for
//     HEIGHT_SHARDED with ROW_MAJOR orientation (contiguous output tile ranges).
struct TilizeMultiCoreShardedRetileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
