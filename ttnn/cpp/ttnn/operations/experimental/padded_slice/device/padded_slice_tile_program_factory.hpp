// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::padded_slice::program {

struct PaddedSliceTileProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
        tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
        tt::tt_metal::KernelHandle untilize_compute_kernel_id = 0;
        ttnn::Shape output_tensor_start;
        ttnn::Shape actual_output_shape;
        tt::tt_metal::CoreCoord compute_with_storage_grid_size;
        uint32_t max_read_size = 0;
        uint32_t max_num_tiles_per_row = 0;
        std::vector<tt::tt_metal::CoreCoord> iter_cores;
        std::tuple<uint32_t, tt::tt_metal::CBHandle> cb_output_tuple;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::padded_slice::program
