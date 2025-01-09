// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/generic_op_device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::generic {

struct GenericOp {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        Tensor& output_tensor,
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
        return ttnn::device_operation::invoke<GenericOpDeviceOperation>(
            queue_id,
            GenericOpDeviceOperation::operation_attributes_t{operation_attributes},
            GenericOpDeviceOperation::tensor_args_t{{input_tensor, output_tensor}});
    }

    static Tensor invoke(
        const Tensor& input_tensor,
        Tensor& output_tensor,
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
        return invoke(0, input_tensor, output_tensor, operation_attributes);
    }

    static Tensor invoke(
        uint8_t queue_id,
        const std::vector<Tensor>& input_tensor,
        Tensor& output_tensor,
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
        std::vector<Tensor> io_tensors = input_tensor;
        io_tensors.push_back(output_tensor);
        return ttnn::device_operation::run<GenericOpDeviceOperation>(
            queue_id,
            GenericOpDeviceOperation::operation_attributes_t{operation_attributes},
            GenericOpDeviceOperation::tensor_args_t{io_tensors});
    }

    static Tensor invoke(
        const std::vector<Tensor>& input_tensors,
        Tensor& output_tensor,
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
        return invoke(0, input_tensors, output_tensor, operation_attributes);
    }
};  // struct UnaryGenericOperation

}   // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", operations::generic::GenericOp>();
}  // namespace ttnn
