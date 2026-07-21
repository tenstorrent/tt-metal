// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Sharded factory for HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED inputs.
// Supports two output paths selected at runtime:
//   - Same-layout sharded L1 output: zero-copy (output CB aliased to shard buffer).
//   - L1 INTERLEAVED output: zero-copy input read, TensorAccessor scatter write.
//     Only valid for HEIGHT_SHARDED with ROW_MAJOR orientation (contiguous output tile ranges).
//     DRAM interleaved output is excluded — DRAM writes always require NoC, so the default factory
//     is used instead (no performance benefit from the optimized path).
struct TilizeMultiCoreShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
