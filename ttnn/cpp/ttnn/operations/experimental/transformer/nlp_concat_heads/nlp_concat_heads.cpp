// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads.hpp"

#include <utility>

namespace ttnn::operations::experimental::nlp_concat_heads {

ttnn::Tensor NLPConcatHeadsOperation::invoke(
    const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::nlp_concat_heads(input_tensor, memory_config);
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads
