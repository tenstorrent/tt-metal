// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

FlipDeviceOperation::MultiCoreTiles::cached_program_t FlipDeviceOperation::MultiCoreTiled::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .core_range = all_cores},
    };
}

void FlipDeviceOperation::MultiCoreTiled::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::data_movement
