// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/kv_cache_load_slice_device_operation.hpp"
#include "kv_cache_load_slice.hpp"

namespace ttnn::experimental {

ttnn::Tensor kv_cache_load_slice(
    const Tensor& input_tensor,
    const uint32_t seq_len_start,
    const uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::kv_cache_load_slice(
        input_tensor, seq_len_start, seq_len_end, memory_config, optional_output_tensor);
}

}  // namespace ttnn::experimental
