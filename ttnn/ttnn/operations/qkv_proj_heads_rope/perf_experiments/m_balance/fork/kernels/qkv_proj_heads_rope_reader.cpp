// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope reader (NCRISC): tile row 0 of the core's bias tiles into cb_bias once, x K-blocks
// into cb_activation_block, then per valid row the rotary table tiles [cos_lo, -sin_lo, sin_hi, cos_hi].
// R3: one core of the M-group reads each x K-block and multicasts it into the same cb_activation_block
// slot on the rest of the group (sender choice is host-side; should_send() covers fixed and rotating).
// Table rows are read in chunks between x rounds (cb_rotary_table holds one row per distinct sequence
// position of a block), so the reads and the sign flip run in the slack while compute works through the
// K-loop.
// After the K-loop the reader stores every out_write_nocs-th output row from cb_out_tiles_noc0 on NoC0
// (rows 1, 1 + out_write_nocs, ...); the writer stores the rest on NoC1.
// The sin sign flip is a tile-content transform no dataflow helper provides.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "qkv_proj_heads_rope_store.hpp"

namespace {

template <uint32_t tile_bytes>
FORCE_INLINE void negate_bf16_tile(uint32_t tile_addr) {
    // Plain (non-volatile) pointer: the page is owned by this RISC between the read barrier and the push,
    // so the compiler may batch the loads.
    uint32_t* words = reinterpret_cast<uint32_t*>(tile_addr);
    constexpr uint32_t num_words = tile_bytes / sizeof(uint32_t);
    constexpr uint32_t unroll = 8;
    static_assert(num_words % unroll == 0);
    for (uint32_t idx = 0; idx < num_words; idx += unroll) {
        uint32_t vals[unroll];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < unroll; ++u) {
            vals[u] = words[idx + u];
        }
#pragma GCC unroll 8
        for (uint32_t u = 0; u < unroll; ++u) {
            words[idx + u] = vals[u] ^ 0x80008000u;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rotary_table = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t k_block = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(8);
    constexpr uint32_t out_write_nocs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_tiles_noc0 = get_compile_time_arg_val(10);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t num_heads = get_compile_time_arg_val(12);
    constexpr uint32_t last_valid_rows = get_compile_time_arg_val(13);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t core_rows = get_arg_val<uint32_t>(4);
    const bool is_rotary = get_arg_val<uint32_t>(5) != 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(6);
    const uint32_t k_addr = get_arg_val<uint32_t>(7);
    const uint32_t v_addr = get_arg_val<uint32_t>(8);
    const uint32_t third = get_arg_val<uint32_t>(9);
    const uint32_t first_head = get_arg_val<uint32_t>(10);
    const uint32_t bias_addr = get_arg_val<uint32_t>(11);
    const uint32_t first_col_tile = get_arg_val<uint32_t>(12);
    constexpr auto x_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xmc_ct_offset"), get_named_compile_time_arg_val("xmc_rt_offset")>();

    // K' = hidden tiles (the bias is seeded into the partials, not a K tile).
    static_assert(num_k_blocks * k_block == hidden_tiles);
    constexpr uint32_t act_block_tiles = block_m * k_block;
    // Table row = [cos_lo, -sin_lo, sin_hi, cos_hi]: the rotate_half tile swap needs D/2 == one tile.
    static_assert(tiles_per_head == 2);
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto cos_accessor = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto sin_accessor = TensorAccessor(sin_args, sin_addr, tile_bytes);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    // Once per core; compute keeps the tiles fronted for every block. The seed is a ROW broadcast, so only
    // tile row 0 is consumed: one 64 B DRAM-aligned span from face 0 and one from face 1 per tile (rows 2..15
    // of the faces stay stale and are never read). Full tiles from all 108 cores at t = 0 burst DRAM ahead of
    // W round 0 (measured: W round 0's multicast moved 4.0 -> 11.6 us). On NoC0, so the writer's W reads
    // never wait on it. Perf 2: the round-0 x sender issues its x reads behind the bias spans under one
    // barrier instead of waiting for the bias barrier first (8x512 x arrival 6.9 -> 6.0 us, -1.0 us e2e).
    bool bias_pending = false;
    {
        MaybeDeviceZoneScope("reader_bias");
        const auto bias_accessor = TensorAccessor(bias_args, bias_addr, tile_bytes);
        constexpr uint32_t face_bytes = 16 * 16 * 2;
        constexpr uint32_t span_bytes = 64;
        cb_reserve_back(cb_bias, n_tiles);
        const uint32_t bias_l1 = get_write_ptr(cb_bias);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            const uint64_t src = bias_accessor.get_noc_addr(first_col_tile + n);
            const uint32_t dst = bias_l1 + n * tile_bytes;
#ifndef ABL_NO_BIAS  // ABLATION-TMP
            noc_async_read(src, dst, span_bytes);
            noc_async_read(src + face_bytes, dst + face_bytes, span_bytes);
#endif
        }
        bias_pending = true;
    }
    auto finish_bias = [&]() {
        if (bias_pending) {
            MaybeDeviceZoneScope("reader_bias_barrier");
            noc_async_read_barrier();
            cb_push_back(cb_bias, n_tiles);
            bias_pending = false;
        }
    };

    Noc noc;
    auto x_sender = x_mcast.sender(noc);
    auto x_receiver = x_mcast.receiver(noc);

    // Sender side of one round: fill this core's slot, then multicast it.
    auto read_x_block = [&](uint32_t k_first, uint32_t rows, uint32_t first_row) {
        const uint32_t block_addr = get_write_ptr(cb_activation_block);
        {
            MaybeDeviceZoneScope("reader_x_read_issue");
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t col = 0; col < k_block; ++col) {
#ifdef ABL_NO_XREAD  // ABLATION-TMP
                    continue;
#endif
                    noc_async_read_page(
                        (first_row + row) * hidden_tiles + k_first + col,
                        x_accessor,
                        block_addr + (row * k_block + col) * tile_bytes);
                }
            }
        }
        {
            MaybeDeviceZoneScope("reader_x_read_barrier");
            noc_async_read_barrier();
        }
        if (bias_pending) {
            cb_push_back(cb_bias, n_tiles);
            bias_pending = false;
        }
        // Valid rows are the leading `rows * k_block` tiles of the slot.
        MaybeDeviceZoneScope("reader_x_mcast_send");
#ifndef ABL_NO_MCAST  // ABLATION-TMP
        x_sender.send(block_addr, block_addr, rows * k_block * tile_bytes);
#endif
    };

    // Table slots [first, last) of the block starting at tile-row first_row; slot j holds sequence
    // position (first_row + j) % seq_tiles (compute reads block row r from slot r % seq_tiles).
    auto read_table_rows = [&](uint32_t first_row, uint32_t first, uint32_t last) {
        if (first == last) {
            return;
        }
        MaybeDeviceZoneScope("reader_table");
        const uint32_t num_tiles = (last - first) * table_tiles_per_row;
        cb_reserve_back(cb_rotary_table, num_tiles);
        uint32_t table_addr = get_write_ptr(cb_rotary_table);
#ifdef ABL_NO_TABLE  // ABLATION-TMP
        cb_push_back(cb_rotary_table, num_tiles);
        return;
#endif
        for (uint32_t row = first; row < last; ++row) {
            const uint32_t lo_page = ((first_row + row) % seq_tiles) * tiles_per_head;
            noc_async_read_page(lo_page, cos_accessor, table_addr);
            noc_async_read_page(lo_page, sin_accessor, table_addr + tile_bytes);
            noc_async_read_page(lo_page + 1, sin_accessor, table_addr + 2 * tile_bytes);
            noc_async_read_page(lo_page + 1, cos_accessor, table_addr + 3 * tile_bytes);
            table_addr += table_tiles_per_row * tile_bytes;
        }
        noc_async_read_barrier();
        table_addr = get_write_ptr(cb_rotary_table);
        for (uint32_t row = first; row < last; ++row) {
#ifndef ABL_NO_NEG  // ABLATION-TMP
            negate_bf16_tile<tile_bytes>(table_addr + tile_bytes);
#endif
            table_addr += table_tiles_per_row * tile_bytes;
        }
        cb_push_back(cb_rotary_table, num_tiles);
    };

    // Every core of an M-group has the same rows, so all walk the same rounds and CB slot sequence.
    uint32_t round = 0;
    if (!(core_rows > 0 && x_mcast.should_send(0))) {
        finish_bias();
    }

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;
        const uint32_t first_row = row_start + block_row;

        // Nominal block quantum keeps the ring aligned; rows past `rows` stay stale and are never consumed.
        // A block's table slots (one cb_rotary_table fill: every block but a core's last fills it exactly,
        // so the slots are contiguous) are spread over the K-blocks, each chunk read right after that
        // K-block's x slot is pushed.
        const uint32_t table_rows = rows < seq_tiles ? rows : seq_tiles;
        uint32_t table_rows_done = 0;
        for (uint32_t k_blk = 0; k_blk < num_k_blocks; ++k_blk, ++round) {
            {
                MaybeDeviceZoneScope("reader_x_reserve");
                cb_reserve_back(cb_activation_block, act_block_tiles);
            }
            if (x_mcast.should_send(round)) {
                read_x_block(k_blk * k_block, rows, first_row);
            } else {
                MaybeDeviceZoneScope("reader_x_receive");
#ifndef ABL_NO_MCAST  // ABLATION-TMP
                x_receiver.receive(round);
#endif
            }
            cb_push_back(cb_activation_block, act_block_tiles);
            if (is_rotary) {
                const uint32_t table_rows_next = (k_blk + 1) * table_rows / num_k_blocks;
                read_table_rows(first_row, table_rows_done, table_rows_next);
                table_rows_done = table_rows_next;
            }
        }

        if constexpr (out_write_nocs > 1) {
            if (rows > 1) {
                // Waiting for the block's first odd row = waiting for the K-loop + first epilogue rows.
                MaybeDeviceZoneScope("reader_store_wait_first");
                cb_wait_front(cb_out_tiles_noc0, n_tiles);
            }
            MaybeDeviceZoneScope("reader_store");
            for (uint32_t row = 1; row < rows; row += out_write_nocs) {
                qkv_store::store_row<
                    cb_out_tiles_noc0,
                    n_tiles,
                    tiles_per_head,
                    num_heads,
                    seq_tiles,
                    tile_bytes,
                    last_valid_rows>(noc, first_row + row, third, first_head, q_accessor, k_accessor, v_accessor);
            }
        }
    }
    finish_bias();
    MaybeDeviceZoneScope("reader_write_barrier");
    noc_async_write_barrier();
}
