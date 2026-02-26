// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct ShardedAdd {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_a, const ttnn::Tensor& input_b);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto s09s2_sharded_add =
    ttnn::register_operation<"ttnn::s09s2_sharded_add", ttnn::operations::onboarding::ShardedAdd>();
}  // namespace ttnn
