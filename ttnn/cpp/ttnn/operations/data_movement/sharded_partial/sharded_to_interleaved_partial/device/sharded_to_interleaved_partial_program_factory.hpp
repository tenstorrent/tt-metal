// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Adapter factory that wraps the shared factory
struct ShardedToInterleavedPartialProgramFactory {
    using shared_variables_t = ShardedToInterleavedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ShardedToInterleavedPartialParams& params,
        const ShardedToInterleavedPartialInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ShardedToInterleavedPartialParams& params,
        const ShardedToInterleavedPartialInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
