// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads.hpp"

#include <utility>

namespace ttnn::experimental {

ttnn::Tensor nlp_concat_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::nlp_concat_heads(input_tensor, memory_config);
}

}  // namespace ttnn::experimental
