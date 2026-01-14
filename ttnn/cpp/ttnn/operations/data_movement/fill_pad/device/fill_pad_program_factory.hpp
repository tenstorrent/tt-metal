// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fill_pad_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::fill_pad::detail {

const std::map<ttnn::DataType, uint32_t> data_type_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT16, 2},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::INT32, 4},
    {ttnn::DataType::UINT8, 1},
};

}  // namespace ttnn::operations::data_movement::fill_pad::detail

namespace ttnn::operations::data_movement::fill_pad::program {

struct FillPadSharedVariables {
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct FillPadProgramFactory {
    using shared_variables_t = FillPadSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::fill_pad::program
