// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct MoeDispatchOffsetsOperation {
    static ttnn::Tensor invoke(const Tensor& input_tensor, uint32_t n_routed_experts);
};

}  // namespace operations::experimental

constexpr auto moe_dispatch_offsets = ttnn::
    register_operation<"ttnn::moe_dispatch_offsets", ttnn::operations::experimental::MoeDispatchOffsetsOperation>();

}  // namespace ttnn
