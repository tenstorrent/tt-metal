// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache {

constexpr uint32_t LATENT_WIDTH = 512;
constexpr uint32_t SCALE_WIDTH = 4;
constexpr uint32_t ROPE_WIDTH = 64;
constexpr uint32_t PACKED_ROW_BYTES = LATENT_WIDTH + SCALE_WIDTH * sizeof(float) + ROPE_WIDTH * sizeof(uint16_t);

ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const Tensor& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache

namespace ttnn::experimental::deepseek_prefill {
using operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::pack_scaled_fp8_kv_cache;
}
