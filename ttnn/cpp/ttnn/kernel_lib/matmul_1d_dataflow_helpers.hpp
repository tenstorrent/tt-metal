// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * read_matmul_tiles: read A and B tiles from DRAM into their CBs for a single-core matmul.
 *
 * Reads tiles in the order consumed by compute_kernel_lib::matmul_1d() with WaitPerTile mode:
 * for each (batch, mt, nt, kt), pushes A[b, mt, kt] then B[b, kt, nt] into their CBs.
 *
 * Uses TensorAccessorArgs chaining so that the program factory can insert compile-time
 * accessor args in the correct order after any named CB compile-time args.
 *
 * PREREQUISITE: Two TensorAccessor compile-time arg blocks must be provided by the program
 * factory, chained in order:
 *   CTA[0 .. N-1]:  accessor args for in0 (s0_args = TensorAccessorArgs<0>())
 *   CTA[N .. M-1]:  accessor args for in1 (s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>())
 *
 * NOTE: This helper produces tiles in the WaitPerTile loop order (batch × Mt × Nt × Kt).
 * It is NOT compatible with matmul_1d WaitUpfront mode, which requires a different tile
 * production order. Use a hand-written reader for WaitUpfront.
 *
 * @tparam in0_cb  Circular buffer index for matrix A tiles (0–31).
 * @tparam in1_cb  Circular buffer index for matrix B tiles (0–31).
 *
 * @param in0_tensor_addr  DRAM byte address of matrix A (tiled, MK layout).
 * @param in1_tensor_addr  DRAM byte address of matrix B (tiled, KN layout).
 * @param Mt               Number of tile rows in A (and C).
 * @param Nt               Number of tile columns in B (and C).
 * @param Kt               Number of inner-dimension tiles.
 * @param batch            Number of independent batch slices (default: 1).
 * @param bcast_B          If true, B is not batched — the same B is used for all batch
 *                         slices. B tile index ignores the batch dimension when true.
 *                         Matches the bcast_B convention in reader_bmm_tile_layout.cpp.
 *
 * ── CB Sizing Requirements ───────────────────────────────────────────────────
 *
 *   in0_cb: tile-sized pages, >= 1 page
 *   in1_cb: tile-sized pages, >= 1 page
 *
 * ── Tile Index Layout ────────────────────────────────────────────────────────
 *
 *   A[b, mt, kt] linear index: b * Mt * Kt + mt * Kt + kt
 *   B[b, kt, nt] linear index: (bcast_B ? 0 : b * Kt * Nt) + kt * Nt + nt
 */
template <uint32_t in0_cb, uint32_t in1_cb>
FORCE_INLINE void read_matmul_tiles(
    uint32_t in0_tensor_addr,
    uint32_t in1_tensor_addr,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t batch = 1,
    bool bcast_B = false) {
    // Set up TensorAccessors using chained compile-time arg offsets.
    // The program factory must insert accessor args in this chained order.
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, in0_tensor_addr, get_tile_size(in0_cb));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, in1_tensor_addr, get_tile_size(in1_cb));

    // Loop order matches matmul_1d() compute: batch × Mt × Nt × Kt.
    // For each (b, mt, nt, kt): push A[b,mt,kt] then B[b,kt,nt].
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    // A tile at (b, mt, kt) — A is [batch, Mt, Kt] in tile layout
                    uint32_t a_tile_index = b * Mt * Kt + mt * Kt + kt;
                    cb_reserve_back(in0_cb, 1);
                    noc_async_read_tile(a_tile_index, s0, get_write_ptr(in0_cb));
                    noc_async_read_barrier();
                    cb_push_back(in0_cb, 1);

                    // B tile at (b, kt, nt) — B is [batch, Kt, Nt] in tile layout.
                    // When bcast_B is true, B is not batched: use b=0 for all batch slices.
                    uint32_t b_tile_index = (bcast_B ? 0 : b * Kt * Nt) + kt * Nt + nt;
                    cb_reserve_back(in1_cb, 1);
                    noc_async_read_tile(b_tile_index, s1, get_write_ptr(in1_cb));
                    noc_async_read_barrier();
                    cb_push_back(in1_cb, 1);
                }
            }
        }
    }
}

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
