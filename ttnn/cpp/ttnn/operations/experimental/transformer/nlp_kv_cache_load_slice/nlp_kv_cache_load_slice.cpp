// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_kv_cache_load_slice_device_operation.hpp"
#include "nlp_kv_cache_load_slice.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor NLPKVCacheLoadSliceOperation::invoke(
    const Tensor& input_tensor,
    const uint32_t seq_len_start,
    const uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::nlp_kv_cache_load_slice(
        input_tensor, seq_len_start, seq_len_end, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::transformer
