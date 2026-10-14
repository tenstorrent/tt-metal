// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache {

constexpr uint32_t LATENT_WIDTH = 512;
constexpr uint32_t SCALE_WIDTH = 4;
constexpr uint32_t ROPE_WIDTH = 64;
constexpr uint32_t PACKED_ROW_BYTES = LATENT_WIDTH + SCALE_WIDTH * sizeof(float) + ROPE_WIDTH * sizeof(uint16_t);

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache
