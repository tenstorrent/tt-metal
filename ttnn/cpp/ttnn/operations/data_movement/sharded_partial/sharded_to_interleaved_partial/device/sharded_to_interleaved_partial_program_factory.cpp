// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp"

namespace ttnn::operations::data_movement::program {

ShardedToInterleavedPartialProgramFactory::cached_program_t ShardedToInterleavedPartialProgramFactory::create(
    const sharded_to_interleaved_partial_operation_attributes_t& operation_attributes,
    const sharded_to_interleaved_partial_tensor_args_t& tensor_args,
    partial_tensor_return_value_t& output) {
    // Convert partial types to shared types
    sharded_to_interleaved_operation_attributes_t shared_attrs{
        .output_mem_config = operation_attributes.output_mem_config,
        .output_dtype = operation_attributes.output_dtype,
        .num_slices = operation_attributes.num_slices,
        .slice_index = operation_attributes.slice_index,
    };

    sharded_to_interleaved_tensor_args_t shared_tensor_args{
        .input_tensor = tensor_args.input_tensor,
        .preallocated_output = tensor_args.cache_tensor,
    };

    // Delegates to shared to interleaved factory
    return ShardedToInterleavedProgramFactory::create(shared_attrs, shared_tensor_args, output);
}

void ShardedToInterleavedPartialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const sharded_to_interleaved_partial_operation_attributes_t& operation_attributes,
    const sharded_to_interleaved_partial_tensor_args_t& tensor_args,
    partial_tensor_return_value_t& output) {
    // Convert partial types to shared types
    sharded_to_interleaved_operation_attributes_t shared_attrs{
        .output_mem_config = operation_attributes.output_mem_config,
        .output_dtype = operation_attributes.output_dtype,
        .num_slices = operation_attributes.num_slices,
        .slice_index = operation_attributes.slice_index,
    };

    sharded_to_interleaved_tensor_args_t shared_tensor_args{
        .input_tensor = tensor_args.input_tensor,
        .preallocated_output = tensor_args.cache_tensor,
    };

    // Delegates to shared to interleaved factory
    ShardedToInterleavedProgramFactory::override_runtime_arguments(
        cached_program, shared_attrs, shared_tensor_args, output);
}

}  // namespace ttnn::operations::data_movement::program
