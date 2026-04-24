// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Single source of truth for Phase-2 V-matmul subblock height. Host-side cb_out sizing
// (streaming_cb_out_tiles) and the kernel (compute_streaming.hpp Phase 2) MUST use the
// same value — a mismatch silently undersizes cb_out and the V matmul packs past
// fifo_limit. constexpr so it can be used in both host code and kernel templates.
namespace ttnn::transformer::sdpa {

constexpr uint32_t streaming_qktv_h(uint32_t subblock_h, uint32_t subblock_w, uint32_t dst_size, uint32_t Sq_chunk_t) {
    // Host subblock solver requires Sq_chunk_t % h == 0, so it falls back to h=1 for odd
    // Sq_chunk_t. The kernel can use h=2 when dest fits 2*w and handle the leftover explicitly.
    return (subblock_h == 1 && 2 * subblock_w <= dst_size && Sq_chunk_t >= 2) ? 2u : subblock_h;
}

}  // namespace ttnn::transformer::sdpa
