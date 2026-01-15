// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "reshape_device_operation_types.hpp"

namespace ttnn::operations::data_movement::reshape_on_device {

struct ReshapeTileProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const reshape_on_device::ReshapeOnDeviceParams& operation_attributes,
        const reshape_on_device::ReshapeOnDeviceInputs& tensor_args,
        reshape_on_device::tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const reshape_on_device::ReshapeOnDeviceParams& operation_attributes,
        const reshape_on_device::ReshapeOnDeviceInputs& tensor_args,
        reshape_on_device::tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::data_movement::reshape_on_device
