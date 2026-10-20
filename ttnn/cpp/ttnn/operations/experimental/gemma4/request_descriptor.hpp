// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::experimental::gemma4::request {

constexpr uint32_t active = 0;
constexpr uint32_t cache_slot = 1;
constexpr uint32_t prefix_tokens = 2;
constexpr uint32_t valid_tokens = 3;
constexpr uint32_t packed_offset_tiles = 4;
constexpr uint32_t packed_span_tiles = 5;
constexpr uint32_t words = 8;
constexpr uint32_t bytes = words * sizeof(uint32_t);

}  // namespace ttnn::experimental::gemma4::request
