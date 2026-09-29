// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache {

// Packs one sparse-SDPA SCALED_FP8 row per token: [latent FP8 bytes | latent/128 FP32 scales | BF16 RoPE], the
// geometry of sparse_sdpa_common.hpp. Widths come from the inputs (latent a multiple of 128). Without ``rope``
// the row ends after the scales (scaled FP8 over every dimension, e.g. DeepSeek-V4.1's 512-dim KV).
ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const std::optional<Tensor>& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache

namespace ttnn::experimental::deepseek_prefill {
using operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::pack_scaled_fp8_kv_cache;
}
