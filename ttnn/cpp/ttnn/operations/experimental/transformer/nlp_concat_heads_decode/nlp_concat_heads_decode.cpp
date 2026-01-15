// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_decode_device_operation.hpp"
#include "nlp_concat_heads_decode.hpp"

#include <utility>

namespace ttnn::operations::experimental::nlp_concat_heads_decode {

ttnn::Tensor NLPConcatHeadsDecodeOperation::invoke(
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::prim::nlp_concat_heads_decode(
        input_tensor, num_heads, memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode
