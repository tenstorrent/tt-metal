// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::padded_slice::program {

struct PaddedSliceRMProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
        tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
        ttnn::Shape output_tensor_start;
        ttnn::Shape actual_output_shape;
        tt::tt_metal::CoreCoord compute_with_storage_grid_size;
        uint32_t max_read_size = 0;
        std::vector<tt::tt_metal::CoreCoord> iter_cores;
        tt::tt_metal::CBHandle cb_output = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PaddedSliceParams& operation_attributes,
        const PaddedSliceInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::padded_slice::program
