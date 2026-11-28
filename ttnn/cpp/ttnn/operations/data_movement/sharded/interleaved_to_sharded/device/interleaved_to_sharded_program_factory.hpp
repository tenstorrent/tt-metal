// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "interleaved_to_sharded_op_types.hpp"

namespace ttnn::prim {

// Per-core L1 scratch bytes for the row-major reader; bank runs are only picked when their stage fits
// `l1_budget`. Shared with the to_memory_config L1 preflight so both budget the same allocation.
uint32_t interleaved_to_sharded_row_major_scratch_size(
    const Tensor& input,
    TensorMemoryLayout shard_strategy,
    uint32_t input_unit_size,
    uint32_t shard_height,
    uint32_t num_cores,
    uint32_t l1_budget);

// Same, budgeted against the L1 free once the output is allocated; part of the program hash.
uint32_t interleaved_to_sharded_row_major_scratch_size(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

struct InterleavedToShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
