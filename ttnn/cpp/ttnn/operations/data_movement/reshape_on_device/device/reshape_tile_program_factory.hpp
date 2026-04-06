// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "reshape_device_operation_types.hpp"

namespace ttnn::prim {

struct ReshapeTileProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::ReshapeOnDeviceParams& operation_attributes,
        const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
        tt::tt_metal::Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::ReshapeOnDeviceParams& operation_attributes,
        const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
        tt::tt_metal::Tensor& output_tensor);
};

}  // namespace ttnn::prim
