// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "interleaved_to_sharded_op_types.hpp"

namespace ttnn::prim {

inline constexpr uint32_t interleaved_to_sharded_num_trids = 4;

struct InterleavedToShardedScratch {
    bool bank_runs = false;
    uint32_t page_stride = 0;
    uint32_t size_per_node = 0;
};

InterleavedToShardedScratch interleaved_to_sharded_row_major_scratch(
    const Tensor& input,
    uint32_t input_unit_size,
    uint32_t num_units_per_shard_height,
    tt::tt_metal::TensorMemoryLayout shard_strategy,
    uint32_t num_cores);

struct InterleavedToShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
