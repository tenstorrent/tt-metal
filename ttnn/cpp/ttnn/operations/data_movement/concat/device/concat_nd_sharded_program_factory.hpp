// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Shared variables for ND sharded concat (inputs and output are block-sharded with NdShardSpec).
// Used to update buffer addresses when the same program is reused with different tensor pointers.
struct ConcatNDShardedSharedVariables {
    uint32_t num_input_tensors = 0;
    std::vector<tt::tt_metal::CBHandle> cb_inputs;
    tt::tt_metal::CBHandle cb_output = 0;
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
};

// Program factory for concat when all input tensors and the output tensor are ND sharded
// (same grid, block-sharded; shard shapes may differ only along the concat dimension).
// Each core holds its shard of each input and writes its shard of the output by
// reading input shards in concat order into a circular buffer, then writing to output.
struct ConcatNDShardedProgramFactory {
    using shared_variables_t = ConcatNDShardedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ConcatParams& operation_attributes,
        const ConcatInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
