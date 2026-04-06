// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp"

namespace ttnn::prim {

ShardedToInterleavedPartialProgramFactory::cached_program_t ShardedToInterleavedPartialProgramFactory::create(
    const ShardedToInterleavedPartialParams& params,
    const ShardedToInterleavedPartialInputs& tensor_args,
    Tensor& output_tensor) {
    // Convert partial types to shared types
    ShardedToInterleavedParams shared_attrs{
        .output_mem_config = params.output_mem_config,
        .output_dtype = params.output_dtype,
        .num_slices = params.num_slices,
        .slice_index = params.slice_index,
    };

    ShardedToInterleavedInputs shared_tensor_args{
        .input_tensor = tensor_args.input_tensor,
        .preallocated_output = tensor_args.cache_tensor,
    };

    // Delegates to shared to interleaved factory
    return ShardedToInterleavedProgramFactory::create(shared_attrs, shared_tensor_args, output_tensor);
}

void ShardedToInterleavedPartialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ShardedToInterleavedPartialParams& params,
    const ShardedToInterleavedPartialInputs& tensor_args,
    Tensor& output_tensor) {
    // Convert partial types to shared types
    ShardedToInterleavedParams shared_attrs{
        .output_mem_config = params.output_mem_config,
        .output_dtype = params.output_dtype,
        .num_slices = params.num_slices,
        .slice_index = params.slice_index,
    };

    ShardedToInterleavedInputs shared_tensor_args{
        .input_tensor = tensor_args.input_tensor,
        .preallocated_output = tensor_args.cache_tensor,
    };

    // Delegates to shared to interleaved factory
    ShardedToInterleavedProgramFactory::override_runtime_arguments(
        cached_program, shared_attrs, shared_tensor_args, output_tensor);
}

}  // namespace ttnn::prim
