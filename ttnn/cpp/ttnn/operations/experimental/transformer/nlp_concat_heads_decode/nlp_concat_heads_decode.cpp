// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_decode_device_operation.hpp"
#include "nlp_concat_heads_decode.hpp"

#include <utility>
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental {

ttnn::Tensor nlp_concat_heads_decode(
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_OP_SCOPE("ttnn::experimental::nlp_concat_heads_decode");
    return ttnn::prim::nlp_concat_heads_decode(
        input_tensor, num_heads, memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace ttnn::experimental
