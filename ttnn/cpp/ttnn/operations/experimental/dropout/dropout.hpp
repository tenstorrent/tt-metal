
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

struct DropoutOperation {
    static Tensor invoke(
        const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed = true);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto dropout =
    ttnn::register_operation<"ttnn::experimental::dropout", ttnn::operations::experimental::DropoutOperation>();
}  // namespace ttnn::experimental
