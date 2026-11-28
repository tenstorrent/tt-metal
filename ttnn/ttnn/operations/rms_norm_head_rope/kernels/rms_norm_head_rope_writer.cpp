// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm_head_rope — writer (BRISC, NoC1).
// prologue (writer is idle at kernel start, so this does not delay the reader's first x reads):
//   column-select tiles for compute's packed-column rsqrt — S_h = 1/HEAD_DIM down column h (cb_colsel),
//   E_h = 1.0 at [h][0] (cb_colext), h < block_heads <= 16 (only faces 0 and 2 are touched). bf16, resident.
// store_block: streamed in chunks of store_chunk_heads heads — wait a chunk, issue its writes, flush, pop — so
// stores start as soon as compute packs a chunk instead of after the whole block. The tail-block padding pages
// compute pushes (block_tiles - valid_tiles) are popped at the end of the block. One write barrier per block.
// Row-tiles come from cb_ctrl_writer (the reader's dynamic claim sequence), ending at the 0xFFFFFFFF marker.
// [WEIGHT_FWD] weight forwarding prologue (Perf 2, measured: 2048x3072 w+rope -9.6%, 2304x3072 weight-only -10.5%,
//   2304x3072 w+rope -2.6%, medians of 3): the row-0 bytes of the head group's weight tiles are read from DRAM ONCE per
//   leader (Rg x fewer 64 B DRAM requests) into a compact staging CB (128 B per tile), pushed with one write per peer
//   to the peer's identical staging address, then flagged with a plain 4 B remote write (NOT noc_semaphore_inc: a
//   non-posted atomic returns its old value into MEM_NOC_ATOMIC_RET_VAL_ADDR — the word the reader's row claim reads —
//   and corrupted the claim ~1 in 20 runs). Every core scatters staging -> row 0/1 of face 0/1 of its cb_weight_row
//   tiles with local NoC reads. No receiver-ready handshake: dispatch sends a program's go signal only after every
//   worker finished the previous program (wait_for_workers_and_send_go_signal), and staging is written by nobody else.
//   Runs on BRISC (idle at start) so the reader's x reads start immediately. Raw dataflow: no kernel_lib helper
//   forwards a buffer to a strided (non-rectangular) core set; the mcast helper needs consecutive receiver groups, and
//   per-column hardware multicast measured 45-73 us leader time under the x-read load.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_heads = get_compile_time_arg_val(1);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t store_chunk_heads = get_compile_time_arg_val(4);
    constexpr uint32_t block_tiles = block_heads * head_tiles;
    constexpr uint32_t cb_colsel = get_compile_time_arg_val(5);
    constexpr uint32_t cb_colext = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ctrl_writer = get_compile_time_arg_val(7);
    constexpr bool dynamic_rows = get_compile_time_arg_val(8) != 0;
    [[maybe_unused]] constexpr uint32_t cb_weight_row = get_compile_time_arg_val(9);
    [[maybe_unused]] constexpr uint32_t cb_w_stage = get_compile_time_arg_val(10);
    [[maybe_unused]] constexpr uint32_t sem_w_ready = get_compile_time_arg_val(11);
    constexpr auto out_args = TensorAccessorArgs<12>();
    [[maybe_unused]] constexpr auto w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t head_start = get_arg_val<uint32_t>(1);
    const uint32_t num_heads = get_arg_val<uint32_t>(2);
    const uint32_t static_row = get_arg_val<uint32_t>(3);  // this core's static first row-tile
    [[maybe_unused]] const uint32_t w_addr = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const bool w_is_leader = get_arg_val<uint32_t>(5) != 0;
    [[maybe_unused]] const uint32_t w_num_peers = get_arg_val<uint32_t>(6);  // leader: (x, y) pairs follow at arg 7

    constexpr uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    DataflowBuffer out_dfb(cb_output_tiles);

    {
        MaybeDeviceZoneScope("writer_constants");
        constexpr uint32_t tb = get_tile_size(cb_colsel);
        DataflowBuffer sel(cb_colsel);
        DataflowBuffer ext(cb_colext);
        sel.reserve_back(block_heads);
        ext.reserve_back(block_heads);
        Noc noc;
        noc.async_write_zeros(sel, block_heads * tb);
        noc.async_write_zeros(ext, block_heads * tb);
        noc.write_zeros_l1_barrier();
        const uint16_t s_bits =
            static_cast<uint16_t>(__builtin_bit_cast(uint32_t, 1.0f / static_cast<float>(head_tiles * 32)) >> 16);
        for (uint32_t h = 0; h < block_heads; ++h) {
            volatile tt_l1_ptr uint16_t* s16 =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sel.get_write_ptr() + h * tb);
            volatile tt_l1_ptr uint16_t* e16 =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ext.get_write_ptr() + h * tb);
            for (uint32_t r = 0; r < 32; ++r) {
                // element (r, h), h < 16: face (r / 16) * 2, offset (r % 16) * 16 + h
                s16[(((r >> 4) * 2) << 8) + ((r & 15) << 4) + h] = s_bits;
            }
            e16[h << 4] = 0x3F80;  // element (h, 0): face 0, offset h * 16
        }
        sel.push_back(block_heads);
        ext.push_back(block_heads);
    }

#if defined(HAS_WEIGHT) && defined(WEIGHT_FWD)
    {
        MaybeDeviceZoneScope("writer_weight");
        constexpr uint32_t w_tile_bytes = get_tile_size(cb_weight_row);
        constexpr uint32_t face_bytes = w_tile_bytes / 4;
        constexpr uint32_t row0_read_bytes = 64;  // DRAM alignment; rows 1..31 of the L1 page are don't-care
        constexpr uint32_t stage_tile_bytes = 2 * row0_read_bytes;
        const uint32_t weight_tiles = num_heads * head_tiles;
        DataflowBuffer wrow_dfb(cb_weight_row);
        DataflowBuffer stage_dfb(cb_w_stage);
        wrow_dfb.reserve_back(weight_tiles);
        stage_dfb.reserve_back(weight_tiles);
        const uint32_t stage_l1 = stage_dfb.get_write_ptr();
        const uint32_t flag_l1 = get_semaphore(sem_w_ready);
        if (w_is_leader) {
            const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
            const uint32_t first = head_start * head_tiles;
            uint32_t s = stage_l1;
            for (uint32_t i = 0; i < weight_tiles; ++i) {
                noc_async_read(w_acc.get_noc_addr(first + i, 0), s, row0_read_bytes);
                noc_async_read(w_acc.get_noc_addr(first + i, face_bytes), s + row0_read_bytes, row0_read_bytes);
                s += stage_tile_bytes;
            }
            noc_async_read_barrier();
            const uint32_t stage_bytes = weight_tiles * stage_tile_bytes;
            for (uint32_t p = 0; p < w_num_peers; ++p) {
                const uint32_t px = get_arg_val<uint32_t>(7 + 2 * p);
                const uint32_t py = get_arg_val<uint32_t>(8 + 2 * p);
                noc_async_write(stage_l1, get_noc_addr(px, py, stage_l1), stage_bytes);
            }
            noc_async_write_barrier();                                     // data lands before the flag
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;  // flag source (a leader never waits on it)
            for (uint32_t p = 0; p < w_num_peers; ++p) {
                const uint32_t px = get_arg_val<uint32_t>(7 + 2 * p);
                const uint32_t py = get_arg_val<uint32_t>(8 + 2 * p);
                noc_semaphore_set_remote(flag_l1, get_noc_addr(px, py, flag_l1));
            }
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1), 1);
        }
        uint32_t s = stage_l1;
        uint32_t l1 = wrow_dfb.get_write_ptr();
        for (uint32_t i = 0; i < weight_tiles; ++i) {
            noc_async_read(get_noc_addr(s), l1, row0_read_bytes);
            noc_async_read(get_noc_addr(s + row0_read_bytes), l1 + face_bytes, row0_read_bytes);
            s += stage_tile_bytes;
            l1 += w_tile_bytes;
        }
        noc_async_read_barrier();
        noc_async_write_barrier();  // leader: flag writes complete
        wrow_dfb.push_back(weight_tiles);
    }
#endif

    const uint32_t blocks_per_row = (num_heads + block_heads - 1) / block_heads;
    DataflowBuffer ctrl(cb_ctrl_writer);
    for (uint32_t r = 0;; ++r) {
        uint32_t row = static_row;
        if constexpr (dynamic_rows) {
            ctrl.wait_front(1);
            row = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl.get_read_ptr());
            ctrl.pop_front(1);
            if (row == 0xFFFFFFFFu) {
                break;
            }
        } else if (r != 0) {
            break;  // Rg >= Rt: exactly one (static) row-tile per core, no ctrl handshake
        }
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint32_t first_head = b * block_heads;
            const uint32_t valid_heads =
                (num_heads - first_head) < block_heads ? (num_heads - first_head) : block_heads;
            const uint32_t valid_tiles = valid_heads * head_tiles;
            const uint32_t first_tile = row * Wt + (head_start + first_head) * head_tiles;

            for (uint32_t h = 0; h < valid_heads; h += store_chunk_heads) {
                const uint32_t chunk_heads =
                    (valid_heads - h) < store_chunk_heads ? (valid_heads - h) : store_chunk_heads;
                const uint32_t chunk_tiles = chunk_heads * head_tiles;
                {
                    MaybeDeviceZoneScope("writer_wait");
                    out_dfb.wait_front(chunk_tiles);
                }
                {
                    MaybeDeviceZoneScope("writer_issue");
                    uint32_t l1 = out_dfb.get_read_ptr();
                    const uint32_t chunk_first = first_tile + h * head_tiles;
                    for (uint32_t i = 0; i < chunk_tiles; ++i) {
                        noc_async_write_page(chunk_first + i, out_acc, l1);
                        l1 += tile_bytes;
                    }
                }
                {
                    MaybeDeviceZoneScope("writer_flush");
                    noc_async_writes_flushed();
                }
                out_dfb.pop_front(chunk_tiles);
            }
            if (valid_tiles < block_tiles) {
                out_dfb.wait_front(block_tiles - valid_tiles);
                out_dfb.pop_front(block_tiles - valid_tiles);
            }
            {
                MaybeDeviceZoneScope("writer_barrier");
                noc_async_write_barrier();
            }
        }
    }
}
