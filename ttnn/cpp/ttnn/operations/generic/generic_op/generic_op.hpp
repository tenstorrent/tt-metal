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
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes,
        const std::vector<Tensor>& io_tensors = {}) {
        return ttnn::prim::generic(queue_id, input_tensor, operation_attributes, io_tensors);
    }

    static Tensor invoke(
        const Tensor& input_tensor,
        const GenericOpDeviceOperation::operation_attributes_t& operation_attributes,
        const std::vector<Tensor>& io_tensors = {}) {
        return invoke(0, input_tensor, operation_attributes, io_tensors);
    }

    // static Tensor invoke(
    //     uint8_t queue_id,
    //     const std::vector<Tensor>& input_tensor,
    //     const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
    //     return ttnn::prim::generic(0, input_tensor, operation_attributes);
    // }

    // static Tensor invoke(
    //     const std::vector<Tensor>& input_tensors,
    //     const GenericOpDeviceOperation::operation_attributes_t& operation_attributes) {
    //     return invoke(0, input_tensors, operation_attributes);
    // }
};  // struct UnaryGenericOperation

}   // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}  // namespace ttnn
