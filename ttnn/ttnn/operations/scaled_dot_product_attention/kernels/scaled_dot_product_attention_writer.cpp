// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention writer (BRISC / NoC1).
//
// Per work unit (b, h, q_chunk): streams cur_cq * Dt output tiles from
// cb_out_tiles (tile-row-major (r, d) order — matches the compute's
// TileRowMajor pack / streaming chain pack) into the interleaved output.
//
// KV-mcast mode (GROUP > 0, perf-juice): the writer also produces cb_v —
// the group leader reads each V block from DRAM and row-mcasts it to the
// followers on NoC1 (semaphores 2/3), splitting KV bandwidth across both
// NoCs (reader carries Q + K^T on NoC0).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t c_q = get_compile_time_arg_val(3);
    constexpr uint32_t Nq = get_compile_time_arg_val(4);
    constexpr uint32_t c_q_last = get_compile_time_arg_val(5);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(6);
    constexpr uint32_t c_kv = get_compile_time_arg_val(7);
    constexpr uint32_t Nkv = get_compile_time_arg_val(8);
    constexpr uint32_t GROUP = get_compile_time_arg_val(9);
    // ONES_COL: each V row carries a leading all-ones tile (PV then also
    // accumulates the softmax denominator l in output column 0).
    constexpr bool ONES_COL = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t V_W = ONES_COL ? Dt + 1 : Dt;

    constexpr auto out_args = TensorAccessorArgs<11>();
    [[maybe_unused]] constexpr auto v_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    uint32_t v_addr = 0, is_leader = 1, first_x = 0, first_y = 0, last_x = 0, last_y = 0, lead_x = 0, lead_y = 0;
    if constexpr (GROUP > 0) {
        v_addr = get_arg_val<uint32_t>(3);
        is_leader = get_arg_val<uint32_t>(4);
        first_x = get_arg_val<uint32_t>(5);
        first_y = get_arg_val<uint32_t>(6);
        last_x = get_arg_val<uint32_t>(7);
        last_y = get_arg_val<uint32_t>(8);
        lead_x = get_arg_val<uint32_t>(9);
        lead_y = get_arg_val<uint32_t>(10);
    } else if constexpr (ONES_COL) {
        v_addr = get_arg_val<uint32_t>(3);  // every core reads its own V
    }

    if (num_units == 0) {
        return;
    }

    constexpr uint32_t cb_out_tiles = 16;
    constexpr uint32_t cb_v_tiles = 2;
    const uint32_t tile_bytes = get_tile_size(cb_out_tiles);
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);
    [[maybe_unused]] const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    if constexpr (ONES_COL) {
        // Fill the ones tiles once per CB region (leader only — the mcast
        // carries them to followers as part of each block). The V loop never
        // overwrites these positions afterwards (rotation is exactly 2 blocks).
        if (is_leader) {
            constexpr uint32_t block_tiles = c_kv * V_W;
            cb_reserve_back(cb_v_tiles, 2 * block_tiles);
            uint32_t base = get_write_ptr(cb_v_tiles);
            constexpr uint16_t ONE_BF16 = 0x3F80;
            for (uint32_t blk = 0; blk < 2; ++blk) {
                for (uint32_t n = 0; n < c_kv; ++n) {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        base + (blk * block_tiles + n * V_W) * tile_bytes);
                    for (uint32_t i = 0; i < 1024; ++i) {
                        p[i] = ONE_BF16;
                    }
                }
            }
        }
    }

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t bh = unit / Nq;
        const uint32_t qc = unit % Nq;
        const uint32_t cur_cq = (qc == Nq - 1) ? c_q_last : c_q;
        const uint32_t q_row0 = qc * c_q;
        const uint32_t head_base = bh * Sq_t * Dt;

        if constexpr (GROUP > 0 || ONES_COL) {
            // Produce all V blocks for this unit (uniform-chunk fast path).
            // GROUP > 0: leader reads + mcasts; GROUP == 0: every core reads.
            const uint32_t kv_head_base = bh * Skv_t * Dt;
            constexpr uint32_t block_tiles = c_kv * V_W;
            for (uint32_t kb = 0; kb < Nkv; ++kb) {
                cb_reserve_back(cb_v_tiles, block_tiles);
                const uint32_t v_l1 = get_write_ptr(cb_v_tiles);
                if (is_leader) {
                    const uint32_t n0 = kb * c_kv;
                    for (uint32_t n = 0; n < c_kv; ++n) {
                        // Row layout: [ones (if ONES_COL), V row d=0..Dt-1].
                        uint32_t l1_addr = v_l1 + (n * V_W + (ONES_COL ? 1 : 0)) * tile_bytes;
                        for (uint32_t d = 0; d < Dt; ++d) {
                            noc_async_read_tile(kv_head_base + (n0 + n) * Dt + d, v_accessor, l1_addr);
                            l1_addr += tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                }
                if constexpr (GROUP > 0) {
                    volatile tt_l1_ptr uint32_t* ready =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
                    volatile tt_l1_ptr uint32_t* landed =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
                    if (is_leader) {
                        noc_semaphore_wait(ready, GROUP - 1);
                        noc_semaphore_set(ready, 0);
                        // NoC1 multicast: rect coordinates run high->low.
                        const uint64_t v_mcast = get_noc_multicast_addr(last_x, last_y, first_x, first_y, v_l1);
                        noc_async_write_multicast(v_l1, v_mcast, block_tiles * tile_bytes, GROUP - 1);
                        const uint64_t landed_mcast =
                            get_noc_multicast_addr(last_x, last_y, first_x, first_y, get_semaphore(3));
                        *landed = 1;
                        noc_semaphore_set_multicast(get_semaphore(3), landed_mcast, GROUP - 1);
                        noc_async_write_barrier();
                    } else {
                        noc_semaphore_inc(get_noc_addr(lead_x, lead_y, get_semaphore(2)), 1);
                        noc_semaphore_wait(landed, 1);
                        noc_semaphore_set(landed, 0);
                    }
                }
                cb_push_back(cb_v_tiles, block_tiles);
            }
        }

        // Batch the whole chunk: one wait, all writes in flight, one barrier,
        // one pop (CB holds 2 * c_q * Dt pages, so a full chunk always fits).
        const uint32_t chunk_tiles = cur_cq * Dt;
        cb_wait_front(cb_out_tiles, chunk_tiles);
        uint32_t l1_addr = get_read_ptr(cb_out_tiles);
        for (uint32_t r = 0; r < cur_cq; ++r) {
            const uint32_t row_base = head_base + (q_row0 + r) * Dt;
            for (uint32_t d = 0; d < Dt; ++d) {
                noc_async_write_tile(row_base + d, out_accessor, l1_addr);
                l1_addr += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_tiles, chunk_tiles);
    }
}
