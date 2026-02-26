// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_add.hpp"
#include "device/sharded_add_device_operation.hpp"

namespace ttnn::operations::onboarding {

Tensor ShardedAdd::invoke(const ttnn::Tensor& input_a, const ttnn::Tensor& input_b) {
    return ttnn::prim::onboarding_sharded_add(input_a, input_b);
}

}  // namespace ttnn::operations::onboarding
