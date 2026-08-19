// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc writer (BRISC) — weight/bias feed + output scatter.
//
// Per (m_block, n_block):
//   1. push this N-slice's bias tiles (when fused),
//   2. push the weight tiles for every K-block, in the (kk, n) order the
//      matmul helper's in1 indexing expects,
//   3. drain cb_out_rm — each pushed group of Nt_b tile-sized pages holds 32
//      row-major output sticks of `out_slice_bytes` bytes — and scatter them
//      into the NHWC row-major output at column offset n_block*out_slice_bytes.
//
// Multi-core: the M-block index space is split across the grid, so the outer
// loop bound is a runtime arg (`num_m_blocks_here`) offset by this core's
// `start_m_block`. Each output stick (= one DRAM page) belongs to exactly one
// m_block, so no two cores ever touch the same page.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t M_total = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t Kb = get_compile_time_arg_val(4);
    constexpr uint32_t Nt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt_b = get_compile_time_arg_val(6);
    constexpr bool fuse_bias = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t out_slice_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t out_row_bytes = get_compile_time_arg_val(9);  // C_out * elem_size

    constexpr auto weight_args = TensorAccessorArgs<10>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_weight_tiles = 1;
    constexpr uint32_t cb_bias_tiles = 2;
    constexpr uint32_t cb_out_rm = 16;

    constexpr uint32_t TILE_H = 32;

    // ---------------- runtime args ----------------
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    // Per-core M-block range (grid split) — runtime, not CT.
    const uint32_t start_m_block = get_arg_val<uint32_t>(3);
    const uint32_t num_m_blocks_here = get_arg_val<uint32_t>(4);

    const auto weights = TensorAccessor(weight_args, weight_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight_tiles);

    // Every m_block is a FULL Mt tile-rows (M is zero-padded up to num_m_blocks*Mt) so
    // every CB transaction stays size-aligned; padding rows are drained but not written.
    for (uint32_t mb = 0; mb < num_m_blocks_here; ++mb) {
        const uint32_t mt_base = (start_m_block + mb) * Mt;

        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            const uint32_t n_base = n_block * Nt_b;
            // N-dim channel mask (Refinement 3): the matmul produces Nt*32
            // output columns but the output stick only owns `out_row_bytes`.
            // Truncate the last N-block's scatter; the dropped columns are the
            // zero tile-padding of the prepared weight/bias.
            const uint32_t col_off_bytes = n_block * out_slice_bytes;
            const uint32_t remaining_bytes = out_row_bytes - col_off_bytes;
            const uint32_t write_bytes = (remaining_bytes < out_slice_bytes) ? remaining_bytes : out_slice_bytes;

            // ---- bias tiles for this N-slice ----
            if constexpr (fuse_bias) {
                const auto bias = TensorAccessor(bias_args, bias_addr);
                const uint32_t bias_tile_bytes = get_tile_size(cb_bias_tiles);
                cb_reserve_back(cb_bias_tiles, Nt_b);
                uint32_t wptr = get_write_ptr(cb_bias_tiles);
                for (uint32_t c = 0; c < Nt_b; ++c) {
                    noc_async_read(bias.get_noc_addr(n_base + c), wptr, bias_tile_bytes);
                    wptr += bias_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_bias_tiles, Nt_b);
            }

            // ---- weight tiles, one K-block at a time ----
            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                const uint32_t kt_base = k_block * Kb;
                cb_reserve_back(cb_weight_tiles, Kb * Nt_b);
                uint32_t wptr = get_write_ptr(cb_weight_tiles);
                for (uint32_t kk = 0; kk < Kb; ++kk) {
                    const uint32_t row = (kt_base + kk) * Nt + n_base;
                    for (uint32_t c = 0; c < Nt_b; ++c) {
                        noc_async_read(weights.get_noc_addr(row + c), wptr, weight_tile_bytes);
                        wptr += weight_tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_weight_tiles, Kb * Nt_b);
            }

            // ---- scatter the untilized output ----
            for (uint32_t t = 0; t < Mt; ++t) {
                cb_wait_front(cb_out_rm, Nt_b);
                const uint32_t rptr = get_read_ptr(cb_out_rm);
                const uint32_t m_row_base = (mt_base + t) * TILE_H;
                for (uint32_t r = 0; r < TILE_H; ++r) {
                    const uint32_t m = m_row_base + r;
                    if (m < M_total) {
                        noc_async_write(rptr + r * out_slice_bytes, out.get_noc_addr(m, col_off_bytes), write_bytes);
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out_rm, Nt_b);
            }
        }
    }
}
