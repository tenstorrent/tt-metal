// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "slice_write_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::slice_write::program {

struct SliceWriteRMShardedInputSharedVariables {
    std::vector<CoreCoord> iter_cores;
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
    uint32_t max_read_size = 0;
    tt::tt_metal::CBHandle input_cb_handle = 0;
};

struct SliceWriteRMShardedInputProgramFactory {
    using shared_variables_t = SliceWriteRMShardedInputSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

struct SliceWriteTiledShardedInputSharedVariables {
    std::vector<CoreCoord> iter_cores;
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
    std::tuple<uint32_t, tt::tt_metal::CBHandle, uint32_t> cb_input_tuple;
};

struct SliceWriteTiledShardedInputProgramFactory {
    using shared_variables_t = SliceWriteTiledShardedInputSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

struct SliceWriteRMInterleavedSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    CoreCoord compute_with_storage_grid_size;
    uint32_t max_read_size = 0;
};

struct SliceWriteRMInterleavedProgramFactory {
    using shared_variables_t = SliceWriteRMInterleavedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::slice_write::program
