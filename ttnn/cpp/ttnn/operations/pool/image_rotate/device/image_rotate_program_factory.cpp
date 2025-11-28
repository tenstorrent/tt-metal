// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::image_rotate {

using namespace tt;
using namespace tt::tt_metal;

ImageRotateDeviceOperation::ProgramFactory::cached_program_t ImageRotateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Stub - to be implemented by ttnn-factory-builder agent
    TT_THROW(
        "ImageRotateDeviceOperation::ProgramFactory::create is not yet implemented. Awaiting Stage 4-6 "
        "implementation.");
}

void ImageRotateDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Stub - to be implemented by ttnn-factory-builder agent
    TT_THROW(
        "ImageRotateDeviceOperation::ProgramFactory::override_runtime_arguments is not yet implemented. Awaiting Stage "
        "4-6 implementation.");
}

}  // namespace ttnn::operations::image_rotate
