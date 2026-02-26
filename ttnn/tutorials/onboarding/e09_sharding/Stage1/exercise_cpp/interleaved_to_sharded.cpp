// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded.hpp"
#include "device/interleaved_to_sharded_device_operation.hpp"

namespace ttnn::operations::onboarding {

Tensor InterleavedToSharded::invoke(const ttnn::Tensor& input, uint32_t shard_strategy) {
    (void)input;
    (void)shard_strategy;
    // TODO: Call ttnn::prim::onboarding_interleaved_to_sharded with input and shard_strategy
    // Hint: cast shard_strategy to TensorMemoryLayout using static_cast
    TT_THROW("Not implemented");
}

}  // namespace ttnn::operations::onboarding
