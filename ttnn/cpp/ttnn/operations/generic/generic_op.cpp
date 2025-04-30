// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

Tensor GenericOp::invoke(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    return ttnn::prim::generic_op(io_tensors, program_descriptor);
}

}  // namespace ttnn::operations::generic
