// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "convert_to_chw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::cnn::to_chw::program {

struct ConvertToCHWProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_in = 0;
        tt::tt_metal::CBHandle cb_out = 0;
        std::vector<CoreCoord> input_cores;
        tt::tt_metal::KernelHandle reader_kernel_id = 0;
        tt::tt_metal::KernelHandle writer_kernel_id = 0;
        tt::tt_metal::KernelHandle compute_kernel_id = 0;
        uint32_t total_tiles_per_core = 0;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::cnn::to_chw::program
