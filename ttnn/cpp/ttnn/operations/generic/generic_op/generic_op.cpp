// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

struct GenericOp {
    static Tensor invoke(
        QueueId queue_id,
        const std::vector<Tensor>& input_tensors,
        const program_attributes_t& program_attributes,
        const std::vector<Tensor>& io_tensors = {}) {
        return ttnn::prim::generic(queue_id, input_tensor, program_attributes, io_tensors);
    }

    static Tensor invoke(
        const std::vector<Tensor>& input_tensors,
        const program_attributes_t& program_attributes,
        const std::vector<Tensor>& io_tensors = {}) {
        return invoke(0, input_tensor, program_attributes, io_tensors);
    }
};  // struct GenericOp

}  // namespace ttnn::operations::generic
