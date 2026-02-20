// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_falcon7b_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPCreateHeadsFalcon7bOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor_q, const std::optional<MemoryConfig>& memory_config);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_create_qkv_heads_falcon7b = ttnn::register_operation<
    "ttnn::experimental::nlp_create_qkv_heads_falcon7b",
    ttnn::operations::experimental::transformer::NLPCreateHeadsFalcon7bOperation>();

}  // namespace experimental

}  // namespace ttnn
