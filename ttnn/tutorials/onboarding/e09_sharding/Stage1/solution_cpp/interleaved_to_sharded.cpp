// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded.hpp"
#include "device/interleaved_to_sharded_device_operation.hpp"

namespace ttnn::operations::onboarding {

Tensor InterleavedToSharded::invoke(const ttnn::Tensor& input, uint32_t shard_strategy) {
    return ttnn::prim::onboarding_interleaved_to_sharded(input, static_cast<TensorMemoryLayout>(shard_strategy));
}

}  // namespace ttnn::operations::onboarding
