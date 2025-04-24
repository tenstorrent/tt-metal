
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_device_operation.hpp"
#include "dropout.hpp"

namespace ttnn::operations::experimental {

Tensor DropoutOperation::invoke(
    const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed) {
    return ttnn::prim::dropout(input_tensor, prob, scale, seed, use_per_device_seed, DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental
