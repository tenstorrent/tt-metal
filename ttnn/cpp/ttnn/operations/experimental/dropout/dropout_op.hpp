
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/dropout_device_operation.hpp"

namespace ttnn::operations::experimental {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct DropoutOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(const Tensor& input_tensor, float prob, float scale, uint32_t seed) {
        return ttnn::prim::dropout(input_tensor, prob, scale, seed, DataType::BFLOAT16);
    }
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto dropout =
    ttnn::register_operation<"ttnn::::experimental::dropout", ttnn::operations::experimental::DropoutOperation>();
}  // namespace ttnn::experimental
