// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_and_split_heads.hpp"

#include "ttnn/operations/experimental/untilize_and_split_heads/device/untilize_and_split_heads_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor untilize_and_split_heads(
    const ttnn::Tensor& input, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::untilize_and_split_heads(input, num_heads, memory_config);
}

}  // namespace ttnn::experimental
