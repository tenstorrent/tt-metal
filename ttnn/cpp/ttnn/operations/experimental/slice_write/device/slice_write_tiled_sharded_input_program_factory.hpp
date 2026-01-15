// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "slice_write_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::slice_write::program {

struct SliceWriteTiledShardedInputSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> iter_cores;
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
    std::tuple<uint32_t, tt::tt_metal::CBHandle> cb_input_tuple;
};

struct SliceWriteTiledShardedInputProgramFactory {
    using shared_variables_t = SliceWriteTiledShardedInputSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SliceWriteParams& operation_attributes,
        const SliceWriteInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::slice_write::program
