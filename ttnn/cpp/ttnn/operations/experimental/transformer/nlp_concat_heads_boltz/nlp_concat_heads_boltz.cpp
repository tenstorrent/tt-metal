// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_boltz_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads_boltz.hpp"

#include <utility>

namespace ttnn::operations::experimental::transformer {
ttnn::Tensor NLPConcatHeadsBoltzOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return ttnn::prim::nlp_concat_heads_boltz(
        input_tensor, memory_config.value_or(input_tensor.memory_config()), std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::experimental::transformer
