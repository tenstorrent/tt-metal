// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute-core writer (RISCV_0 / NoC1). Block ops:
//   s1_handoff_block  (stage-1 cores) per q block, once compute has packed it into cb_q_slice:
//                       remote quarter -> one block-ready increment on every port of the destination's direction
//                                         (the ports pull the head-rows from my cb_q_slice themselves)
//                       own quarter    -> write my columns of each head-row into the destination core's landing
//                                         scratch (same address on every core), then increment its arrive[p] by
//                                         the tiles written (after the write barrier)
//   store_row_block   per (head, row): prefix N-blocks + rope tiles -> output pages (TILE), one barrier per block

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/q_stem/kernels/q_stem_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_q_slice = get_compile_time_arg_val(0);
    constexpr uint32_t cb_prefix_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rope_out = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t N_t = get_compile_time_arg_val(6);
    constexpr uint32_t Dr_t = get_compile_time_arg_val(7);
    constexpr uint32_t s3n = get_compile_time_arg_val(8);
    constexpr uint32_t n_nb = get_compile_time_arg_val(9);
    constexpr uint32_t n_ports = get_compile_time_arg_val(10);
    constexpr uint32_t sem_block_ready_id = get_compile_time_arg_val(11);
    constexpr uint32_t B = get_compile_time_arg_val(12);
    constexpr uint32_t qrt = get_compile_time_arg_val(13);
    constexpr uint32_t m = get_compile_time_arg_val(14);
    constexpr uint32_t G = get_compile_time_arg_val(15);
    constexpr uint32_t hw = get_compile_time_arg_val(16);
    constexpr auto out_args = TensorAccessorArgs<17>();
    constexpr uint32_t out_w = N_t + Dr_t;
    constexpr uint32_t rows_per_head = B * qrt;

    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t cols = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrive_p_addr = get_arg_val<uint32_t>(a++);
    uint32_t dirmap[G];
    for (uint32_t i = 0; i < G; ++i) {
        dirmap[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t ports_idx = a;
    a += 2 * n_ports;
    const uint32_t n_pieces = get_arg_val<uint32_t>(a++);
    const uint32_t pieces_idx = a;
    a += 4 * n_pieces;
    const uint32_t hpc = get_arg_val<uint32_t>(a++);
    const uint32_t heads_idx = a;

    // ---- stage 1 hand-off -------------------------------------------------------------------------------------------
    if (cols > 0) {
        const uint32_t q_base = get_read_ptr(cb_q_slice);  // never popped before the end: block j at j*m*cols tiles
        const uint32_t block_ready = get_semaphore(sem_block_ready_id);
        q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t j, uint32_t b, uint32_t d, uint32_t r0) {
            cb_wait_front(cb_q_slice, (j + 1) * m * cols);
            if (d != p) {
                for (uint32_t k = 0; k < n_ports; ++k) {
                    if (get_arg_val<uint32_t>(ports_idx + 2 * k + 1) == dirmap[d]) {
                        const uint32_t pc = get_arg_val<uint32_t>(ports_idx + 2 * k);
                        noc_semaphore_inc(get_noc_addr(q_stem::unpack_x(pc), q_stem::unpack_y(pc), block_ready), 1);
                    }
                }
                return;
            }
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t row = b * qrt + r0 + i;
                const uint32_t src_row = q_base + (j * m + i) * cols * tile_bytes;
                for (uint32_t e = 0; e < n_pieces; ++e) {
                    const uint32_t t = pieces_idx + 4 * e;
                    const uint32_t lc = get_arg_val<uint32_t>(t);
                    const uint32_t n = get_arg_val<uint32_t>(t + 1);
                    const uint32_t dc = get_arg_val<uint32_t>(t + 2);
                    const uint32_t slot = get_arg_val<uint32_t>(t + 3) + row * hw;
                    noc_async_write(
                        src_row + lc * tile_bytes,
                        get_noc_addr(q_stem::unpack_x(dc), q_stem::unpack_y(dc), landing_addr + slot * tile_bytes),
                        n * tile_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t e = 0; e < n_pieces; ++e) {
                const uint32_t t = pieces_idx + 4 * e;
                const uint32_t dc = get_arg_val<uint32_t>(t + 2);
                noc_semaphore_inc(
                    get_noc_addr(q_stem::unpack_x(dc), q_stem::unpack_y(dc), arrive_p_addr),
                    m * get_arg_val<uint32_t>(t + 1));
            }
        });
        noc_async_atomic_barrier();
        cb_pop_front(cb_q_slice, num_blocks * m * cols);
    }

    // ---- stage 3 output ---------------------------------------------------------------------------------------------
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t u = 0; u < hpc; ++u) {
        const uint32_t h = get_arg_val<uint32_t>(heads_idx + u);
        for (uint32_t t = 0; t < rows_per_head; ++t) {
            const uint32_t b = t / qrt;
            const uint32_t r = t - b * qrt;
            const uint32_t page0 = ((b * H + h) * qrt + r) * out_w;
            for (uint32_t nb = 0; nb < n_nb; ++nb) {
                cb_wait_front(cb_prefix_out, s3n);
                uint32_t l1 = get_read_ptr(cb_prefix_out);
                for (uint32_t c = 0; c < s3n; ++c) {
                    noc_async_write(l1, out.get_noc_addr(page0 + nb * s3n + c), tile_bytes);
                    l1 += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_prefix_out, s3n);
            }
            cb_wait_front(cb_rope_out, Dr_t);
            uint32_t l1 = get_read_ptr(cb_rope_out);
            for (uint32_t c = 0; c < Dr_t; ++c) {
                noc_async_write(l1, out.get_noc_addr(page0 + N_t + c), tile_bytes);
                l1 += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_rope_out, Dr_t);
        }
    }
}
