// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"

namespace ttnn::operations::experimental::pool {

AvgPool2D::MultiCore::cached_program_t AvgPool2D::MultiCore::create(
    const AvgPool2D::operation_attributes_t& operation_attributes,
    const AvgPool2D::tensor_args_t& tensor_args,
    AvgPool2D::tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    return {std::move(program), {}};
}

void AvgPool2D::MultiCore::override_runtime_arguments(
    AvgPool2D::MultiCore::cached_program_t& cached_program,
    const AvgPool2D::operation_attributes_t& operation_attributes,
    const AvgPool2D::tensor_args_t& tensor_args,
    AvgPool2D::tensor_return_value_t& output_tensor) {}

AvgPool2D::program_factory_t AvgPool2D::select_program_factory(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    return MultiCore{};
}

} // namespace ttnn::operations::experimental::pool
