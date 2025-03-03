
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_device_operation.hpp"
#include "dropout.hpp"
namespace ttnn::operations::experimental {

Tensor DropoutOperation::invoke(
    const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed) {
    auto chip_id = static_cast<uint32_t>(input_tensor.device()->id());
    return ttnn::prim::dropout(
        input_tensor, prob, scale, use_per_device_seed ? seed + chip_id : seed, DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental
