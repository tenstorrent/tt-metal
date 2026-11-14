// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_kv_cache_load_slice_device_operation.hpp"
#include "nlp_kv_cache_load_slice.hpp"

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttnn::experimental {

ttnn::Tensor nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    const uint32_t seq_len_start,
    const uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto output = ttnn::prim::nlp_kv_cache_load_slice(
        input_tensor, seq_len_start, seq_len_end, memory_config, optional_output_tensor);
    // The prim always builds an L1 height-sharded output (its kernel writes shards), so the
    // documented memory_config keyword was silently dropped. Materialize the request on the result;
    // a preallocated output already fixes the configuration.
    if (memory_config.has_value() && !optional_output_tensor.has_value()) {
        output = ttnn::to_memory_config(output, *memory_config);
    }
    return output;
}

}  // namespace ttnn::experimental
