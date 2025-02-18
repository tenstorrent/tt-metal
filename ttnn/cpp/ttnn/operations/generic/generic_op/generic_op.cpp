// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

struct GenericOp {
    static Tensor invoke(
        QueueId queue_id,
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
};  // struct GenericOp

}  // namespace ttnn::operations::generic
