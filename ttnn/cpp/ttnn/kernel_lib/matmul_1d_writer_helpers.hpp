// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * write_matmul_tiles: write C output tiles from a CB to DRAM for a single-core matmul.
 *
 * Reads tiles from out_cb in the order produced by compute_kernel_lib::matmul_1d():
 * for each (batch, mt, nt), pops one output tile and writes it to DRAM at the
 * corresponding linear index.
 *
 * Uses a TensorAccessor with a single compile-time arg block at offset 0.
 *
 * PREREQUISITE: One TensorAccessor compile-time arg block must be provided by the program
 * factory at CTA[0 .. N-1] (s_args = TensorAccessorArgs<0>()).
 *
 * NOTE: Uses noc_async_write_barrier() per tile for simplicity and guaranteed correctness.
 * For higher throughput, noc_async_write_flushed() per tile followed by a single
 * noc_async_write_barrier() after the loop is a valid optimization, but the per-tile
 * barrier form is preferred here.
 *
 * @tparam out_cb  Circular buffer index for matrix C output tiles (0–31).
 *
 * @param out_tensor_addr  DRAM byte address of output matrix C (tiled, MN layout).
 * @param Mt               Number of tile rows in C.
 * @param Nt               Number of tile columns in C.
 * @param batch            Number of independent batch slices (default: 1).
 *
 * ── CB Sizing Requirements ───────────────────────────────────────────────────
 *
 *   out_cb: tile-sized pages, >= 1 page
 *
 * ── Tile Index Layout ────────────────────────────────────────────────────────
 *
 *   C[b, mt, nt] linear index: b * Mt * Nt + mt * Nt + nt
 */
template <uint32_t out_cb>
FORCE_INLINE void write_matmul_tiles(uint32_t out_tensor_addr, uint32_t Mt, uint32_t Nt, uint32_t batch = 1) {
    // Set up TensorAccessor for the output buffer.
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, out_tensor_addr, get_tile_size(out_cb));

    // Loop order matches matmul_1d() output: batch × Mt × Nt.
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                uint32_t tile_index = b * Mt * Nt + mt * Nt + nt;
                cb_wait_front(out_cb, 1);
                noc_async_write_tile(tile_index, s, get_read_ptr(out_cb));
                noc_async_write_barrier();
                cb_pop_front(out_cb, 1);
            }
        }
    }
}

}  // namespace dataflow_kernel_lib
