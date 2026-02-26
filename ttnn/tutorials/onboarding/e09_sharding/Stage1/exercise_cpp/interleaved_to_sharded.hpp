// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct InterleavedToSharded {
    static ttnn::Tensor invoke(const ttnn::Tensor& input, uint32_t shard_strategy);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto s09s1_interleaved_to_sharded = ttnn::
    register_operation<"ttnn::s09s1_interleaved_to_sharded", ttnn::operations::onboarding::InterleavedToSharded>();
}  // namespace ttnn
