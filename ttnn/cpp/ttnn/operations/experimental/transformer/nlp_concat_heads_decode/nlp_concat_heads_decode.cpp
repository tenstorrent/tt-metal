// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_decode_device_operation.hpp"
#include "nlp_concat_heads_decode.hpp"

#include <utility>

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttnn::experimental {

ttnn::Tensor nlp_concat_heads_decode(
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto output = ttnn::prim::nlp_concat_heads_decode(
        input_tensor, num_heads, memory_config, optional_output_tensor, sub_core_grids);
    // The prim takes memory_config as `/*memory_config*/` and always returns an L1 width-sharded
    // tensor, so the documented keyword was silently dropped. Materialize the request on the result
    // (values unchanged); a preallocated output already fixes the configuration.
    if (memory_config.has_value() && !optional_output_tensor.has_value()) {
        output = ttnn::to_memory_config(output, *memory_config);
    }
    return output;
}

}  // namespace ttnn::experimental
