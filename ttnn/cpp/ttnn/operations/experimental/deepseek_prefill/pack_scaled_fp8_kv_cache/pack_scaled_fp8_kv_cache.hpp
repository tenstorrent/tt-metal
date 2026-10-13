// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "packed_kv_layout.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache {

ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const Tensor& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache

namespace ttnn::experimental::deepseek_prefill {
using operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::pack_scaled_fp8_kv_cache;
}
