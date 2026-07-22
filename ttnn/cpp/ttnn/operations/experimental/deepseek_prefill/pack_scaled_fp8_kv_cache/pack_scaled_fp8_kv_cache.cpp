// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_scaled_fp8_kv_cache.hpp"

#include "device/pack_scaled_fp8_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache {

ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const Tensor& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    return ttnn::prim::pack_scaled_fp8_kv_cache(latent, scales, rope, output_memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache
