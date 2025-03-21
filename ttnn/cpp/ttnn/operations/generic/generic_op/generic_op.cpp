// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

Tensor GenericOp::invoke(
    const std::vector<Tensor>& input_tensors,
    // const Tensor& input_tensors,
    const program_attributes_t& program_attributes,
    const std::optional<const Tensor>& output_tensor) {
    return ttnn::prim::generic_op(input_tensors, program_attributes, output_tensor);
}

}  // namespace ttnn::operations::generic
