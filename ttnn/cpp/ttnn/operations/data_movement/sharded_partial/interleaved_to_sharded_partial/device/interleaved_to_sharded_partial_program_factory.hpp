// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "interleaved_to_sharded_partial_op_types.hpp"

namespace ttnn::operations::data_movement::detail {

struct InterleavedToShardedPartialProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id;
        tt::tt_metal::KernelHandle unary_writer_kernel_id;
        tt::tt_metal::CBHandle cb_output;
        std::vector<CoreCoord> cores;
        uint32_t num_slices;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const InterleavedToShardedPartialOperationTypes::operation_attributes_t& operation_attributes,
        const InterleavedToShardedPartialOperationTypes::tensor_args_t& tensor_args,
        InterleavedToShardedPartialOperationTypes::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const InterleavedToShardedPartialOperationTypes::operation_attributes_t& operation_attributes,
        const InterleavedToShardedPartialOperationTypes::tensor_args_t& tensor_args,
        InterleavedToShardedPartialOperationTypes::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::detail
