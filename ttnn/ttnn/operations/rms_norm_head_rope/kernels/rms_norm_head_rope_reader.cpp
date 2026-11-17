// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm_head_rope — reader (NCRISC, NoC0).
//
// prologue_load_constants: [HAS_ROPE] rotation matrix T generated in L1; [HAS_WEIGHT && !WEIGHT_FWD] row 0 of each of
//   this core's weight tiles (two 64 B reads per tile, one barrier). With WEIGHT_FWD the writer owns cb_weight_row.
// Row-tile schedule (dynamic, Perf 1): a core's first row-tile is static (its row-group index); every further
//   row-tile of its head group is claimed from a per-group fetch-and-add counter (program semaphore SEM_ROW_COUNTER
//   on the group's first core, initial value Rg), and only once x space for it is reserved. Why: DRAM reads are not
//   served fairly — on Blackhole NoC0 serves cores in core-y order and the top rows waited ~60 us for their first
//   byte at 120 cores — so a static row split made the starved cores finish ~60 us + their full compute late. With
//   claiming, cores the NoC serves first take more row-tiles. Each claimed row (or the end marker 0xFFFFFFFF) is
//   published to compute and writer through cb_ctrl_compute / cb_ctrl_writer. When Rg >= Rt (dynamic_rows = 0)
//   every row is somebody's static row and the claim round-trip is skipped.
//   The host also swaps the NoCs of the top grid rows (reads on NOC_1) on grids tall enough to starve them.
// load_block (per block = 1 row-tile x block_heads heads x head_tiles tiles):
//   [HAS_ROPE] cos/sin head_tiles tiles each, once per row-tile (pushed with the row-tile's first block);
//   x: valid_heads*head_tiles tile reads (barrier every read_chunk_tiles: caps this core's outstanding reads),
//   push of the FULL block_tiles window (tail blocks push unused pages too, so every push is block-sized and the
//   CB ring never wraps mid-window).

// cos/sin teams (COSSIN_TEAM: HAS_ROPE and team size > 1):
//   Up to COSSIN_TEAM_MAX consecutive head-group cores of a row-group form a team (leader + peers) that processes the
//   SAME row-tile sequence. Only the leader claims rows (one counter per team index, shared by all row-groups) and
//   reads cos/sin from DRAM; it forwards the claimed row to the peers immediately (mailbox semaphore, so peers start
//   their x reads at once) and cos/sin as soon as they land (mcast to the contiguous peer rectangle, else unicast),
//   into the SAME CB slot on each peer — CB write pointers stay in lockstep because every team member pushes the same
//   per-row sequence. Flow control: peers +1 the leader's PEER_READY after reserving x/cos/sin space for the next row;
//   the leader claims only after all G-1 peers are ready. cos/sin data -> ROPE_VALID ordering: same NoC, same VC. No
//   helper covers this (no dataflow work-queue / team-forward helper); raw NoC + semaphores.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_weight_row = get_compile_time_arg_val(3);
    constexpr uint32_t cb_trans_mat = get_compile_time_arg_val(4);
    constexpr uint32_t block_heads = get_compile_time_arg_val(5);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t read_chunk_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t cb_ctrl_compute = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ctrl_writer = get_compile_time_arg_val(10);
    constexpr uint32_t sem_row_counter = get_compile_time_arg_val(11);
    constexpr bool dynamic_rows = get_compile_time_arg_val(12) != 0;  // Rg < Rt: rows beyond the static ones exist
    [[maybe_unused]] constexpr uint32_t sem_peer_ready = get_compile_time_arg_val(13);
    [[maybe_unused]] constexpr uint32_t sem_row_mailbox = get_compile_time_arg_val(14);
    [[maybe_unused]] constexpr uint32_t sem_rope_valid = get_compile_time_arg_val(15);
    constexpr uint32_t block_tiles = block_heads * head_tiles;
    constexpr uint32_t ROW_END = 0xFFFFFFFFu;

    constexpr auto x_args = TensorAccessorArgs<16>();
    [[maybe_unused]] constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto w_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);   // static first row-tile of this core
    const uint32_t total_rows = get_arg_val<uint32_t>(5);  // Rt: rows handed out per head group
    const uint32_t head_start = get_arg_val<uint32_t>(6);
    const uint32_t num_heads = get_arg_val<uint32_t>(7);
    const uint32_t counter_noc_x = get_arg_val<uint32_t>(8);
    const uint32_t counter_noc_y = get_arg_val<uint32_t>(9);
#ifdef COSSIN_TEAM
    const bool is_leader = get_arg_val<uint32_t>(10) != 0;
    const uint32_t leader_x = get_arg_val<uint32_t>(11);
    const uint32_t leader_y = get_arg_val<uint32_t>(12);
    const uint32_t num_peers = get_arg_val<uint32_t>(13);
    const uint32_t mcast_mode = get_arg_val<uint32_t>(14);  // 0 unicast, 1 mcast to the peer row segment
    uint32_t mc_x0 = get_arg_val<uint32_t>(15);
    uint32_t mc_y0 = get_arg_val<uint32_t>(16);
    uint32_t mc_x1 = get_arg_val<uint32_t>(17);
    uint32_t mc_y1 = get_arg_val<uint32_t>(18);
    constexpr uint32_t peer_arg_base = 19;  // leader only: num_peers (x, y) pairs
    if (noc_index == 1) {                   // NoC1 multicast rectangles run end -> start
        uint32_t t = mc_x0;
        mc_x0 = mc_x1;
        mc_x1 = t;
        t = mc_y0;
        mc_y0 = mc_y1;
        mc_y1 = t;
    }
    const uint32_t ready_l1 = get_semaphore(sem_peer_ready);
    const uint32_t mailbox_l1 = get_semaphore(sem_row_mailbox);
    const uint32_t valid_l1 = get_semaphore(sem_rope_valid);
    volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_l1);
    volatile tt_l1_ptr uint32_t* mailbox_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_l1);
    volatile tt_l1_ptr uint32_t* valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1);
    if (is_leader) {
        *valid_ptr = 1;  // leader's own ROPE_VALID word is the source of the "1" it multicasts to peers
    }
    const uint64_t leader_ready_addr = get_noc_addr(leader_x, leader_y, ready_l1);
#endif

    constexpr uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto x_acc = TensorAccessor(x_args, x_addr, tile_bytes);

#ifdef HAS_ROPE
    constexpr uint32_t rope_tile_bytes = get_tile_size(cb_rope_cos);
    const auto cos_acc = TensorAccessor(cos_args, cos_addr, rope_tile_bytes);
    const auto sin_acc = TensorAccessor(sin_args, sin_addr, rope_tile_bytes);
    DataflowBuffer cos_dfb(cb_rope_cos);
    DataflowBuffer sin_dfb(cb_rope_sin);
#endif

#if defined(HAS_WEIGHT) && !defined(WEIGHT_FWD)
    DataflowBuffer wrow_dfb(cb_weight_row);
    const uint32_t weight_tiles = num_heads * head_tiles;
    auto load_weight_rows = [&]() {
        MaybeDeviceZoneScope("reader_weight");
        {
            // Row 0 of a tile = face0 row0 (bytes 0..31) + face1 row0 (bytes 512..543). Read 64 B at each
            // (DRAM alignment); rows 1..31 of the L1 page are don't-care (consumer is a Row broadcast).
            constexpr uint32_t w_tile_bytes = get_tile_size(cb_weight_row);
            constexpr uint32_t face_bytes = w_tile_bytes / 4;
            constexpr uint32_t row0_read_bytes = 64;
            const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
            wrow_dfb.reserve_back(weight_tiles);
            uint32_t l1 = wrow_dfb.get_write_ptr();
            const uint32_t first = head_start * head_tiles;
            for (uint32_t i = 0; i < weight_tiles; ++i) {
                noc_async_read(w_acc.get_noc_addr(first + i, 0), l1, row0_read_bytes);
                noc_async_read(w_acc.get_noc_addr(first + i, face_bytes), l1 + face_bytes, row0_read_bytes);
                l1 += w_tile_bytes;
            }
            noc_async_read_barrier();
            wrow_dfb.push_back(weight_tiles);
        }
    };
#endif

    // ---------------------------------------------------------------- prologue_load_constants
    {
        MaybeDeviceZoneScope("reader_constants");
#ifdef HAS_ROPE
        {
            // T: y @ T = (-y1, y0, -y3, y2, ...). T[2i][2i+1] = +1, T[2i+1][2i] = -1, in tile-face order.
            DataflowBuffer trans_dfb(cb_trans_mat);
            trans_dfb.reserve_back(1);
            Noc noc;
            noc.async_write_zeros(trans_dfb, get_tile_size(cb_trans_mat));
            noc.write_zeros_l1_barrier();
            volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(trans_dfb.get_write_ptr());
            auto at = [](uint32_t r, uint32_t c) -> uint32_t {
                return (((r >> 4) * 2 + (c >> 4)) << 8) + ((r & 15) << 4) + (c & 15);
            };
            for (uint32_t i = 0; i < 16; ++i) {
                t[at(2 * i, 2 * i + 1)] = 0x3F80;  // +1.0 bf16
                t[at(2 * i + 1, 2 * i)] = 0xBF80;  // -1.0 bf16
            }
            trans_dfb.push_back(1);
        }
#endif
    }

#if defined(HAS_WEIGHT) && !defined(WEIGHT_FWD)
    load_weight_rows();
#endif
    // ---------------------------------------------------------------- load_block loop
    DataflowBuffer x_dfb(cb_input_tiles);
    DataflowBuffer ctrl_compute(cb_ctrl_compute);
    DataflowBuffer ctrl_writer(cb_ctrl_writer);
    const uint64_t counter_addr = get_noc_addr(counter_noc_x, counter_noc_y, get_semaphore(sem_row_counter));
    volatile tt_l1_ptr uint32_t* atomic_ret =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    const uint32_t blocks_per_row = (num_heads + block_heads - 1) / block_heads;
    for (uint32_t r = 0;; ++r) {
        if constexpr (!dynamic_rows) {
            // Carve-out (Rg >= Rt): every row-tile is some core's static first row — nothing to claim, so the claim
            // and the ctrl-CB handshake are compiled out (they measured +3-4% on such grids, e.g. 128x512 w+rope).
            if (r != 0) {
                break;
            }
        }
        {
            MaybeDeviceZoneScope("reader_reserve_x");
            x_dfb.reserve_back(block_tiles);  // claim only with space for the row's first block
        }
        uint32_t row = row_start;
#ifdef COSSIN_TEAM
        {
            MaybeDeviceZoneScope("reader_reserve_rope");
            cos_dfb.reserve_back(head_tiles);
            sin_dfb.reserve_back(head_tiles);
        }
        uint32_t ctrl_word = row;
        if (r != 0) {
            MaybeDeviceZoneScope("reader_claim");
            if (is_leader) {
                noc_semaphore_wait(ready_ptr, num_peers);  // every peer has x/cos/sin space for this row
                *ready_ptr = 0;
                noc_semaphore_inc(counter_addr, 1);
                noc_async_atomic_barrier();
                invalidate_l1_cache();
                row = *atomic_ret;
                ctrl_word = row < total_rows ? row : ROW_END;
                const uint32_t mail = ctrl_word == ROW_END ? ROW_END : ctrl_word + 1;
                for (uint32_t p = 0; p < num_peers; ++p) {
                    const uint32_t px = get_arg_val<uint32_t>(peer_arg_base + 2 * p);
                    const uint32_t py = get_arg_val<uint32_t>(peer_arg_base + 2 * p + 1);
                    noc_inline_dw_write(get_noc_addr(px, py, mailbox_l1), mail);
                }
            } else {
                noc_semaphore_inc(leader_ready_addr, 1);
                noc_semaphore_wait_min(mailbox_ptr, 1);
                const uint32_t mail = *mailbox_ptr;
                *mailbox_ptr = 0;
                ctrl_word = mail == ROW_END ? ROW_END : mail - 1;
                row = ctrl_word;
            }
        }
#else
        if (r != 0) {
            MaybeDeviceZoneScope("reader_claim");
            noc_semaphore_inc(counter_addr, 1);  // non-posted INCR_GET: old value lands in MEM_NOC_ATOMIC_RET_VAL_ADDR
            noc_async_atomic_barrier();
            invalidate_l1_cache();
            row = *atomic_ret;
        }
        const uint32_t ctrl_word = row < total_rows ? row : ROW_END;
#endif
        if constexpr (dynamic_rows) {
            ctrl_compute.reserve_back(1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_compute.get_write_ptr()) = ctrl_word;
            ctrl_compute.push_back(1);
            ctrl_writer.reserve_back(1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_writer.get_write_ptr()) = ctrl_word;
            ctrl_writer.push_back(1);
        }
        if (ctrl_word == ROW_END) {
            break;
        }
#ifdef COSSIN_TEAM
        const uint32_t cos_l1 = cos_dfb.get_write_ptr();
        const uint32_t sin_l1 = sin_dfb.get_write_ptr();
        bool rope_pending = is_leader;  // leader: forward once its cos/sin reads land
        auto forward_rope = [&]() {
            MaybeDeviceZoneScope("reader_rope_forward");
            constexpr uint32_t rope_bytes = head_tiles * rope_tile_bytes;
            if (mcast_mode == 1) {
                noc_async_write_multicast(
                    cos_l1, get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, cos_l1), rope_bytes, num_peers);
                noc_async_write_multicast(
                    sin_l1, get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sin_l1), rope_bytes, num_peers);
                noc_semaphore_set_multicast(
                    valid_l1, get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, valid_l1), num_peers);
            } else {
                for (uint32_t p = 0; p < num_peers; ++p) {
                    const uint32_t px = get_arg_val<uint32_t>(peer_arg_base + 2 * p);
                    const uint32_t py = get_arg_val<uint32_t>(peer_arg_base + 2 * p + 1);
                    noc_async_write(cos_l1, get_noc_addr(px, py, cos_l1), rope_bytes);
                    noc_async_write(sin_l1, get_noc_addr(px, py, sin_l1), rope_bytes);
                    noc_semaphore_set_remote(valid_l1, get_noc_addr(px, py, valid_l1));  // same VC: after data
                }
            }
            rope_pending = false;
        };
        if (is_leader) {
            MaybeDeviceZoneScope("reader_issue_rope_lead");
            noc_async_writes_flushed();  // previous forwards have left this core before the slot is refilled
            uint32_t cl = cos_l1;
            uint32_t sl = sin_l1;
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_read_page(row * head_tiles + d, cos_acc, cl);
                noc_async_read_page(row * head_tiles + d, sin_acc, sl);
                cl += rope_tile_bytes;
                sl += rope_tile_bytes;
            }
        }
#elif defined(HAS_ROPE)
        {
            MaybeDeviceZoneScope("reader_reserve_rope");
            cos_dfb.reserve_back(head_tiles);
            sin_dfb.reserve_back(head_tiles);
        }
        {
            MaybeDeviceZoneScope("reader_issue_rope");
            uint32_t cl = cos_dfb.get_write_ptr();
            uint32_t sl = sin_dfb.get_write_ptr();
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_read_page(row * head_tiles + d, cos_acc, cl);
                noc_async_read_page(row * head_tiles + d, sin_acc, sl);
                cl += rope_tile_bytes;
                sl += rope_tile_bytes;
            }
        }
#endif
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint32_t first_head = b * block_heads;
            const uint32_t valid_heads =
                (num_heads - first_head) < block_heads ? (num_heads - first_head) : block_heads;
            const uint32_t valid_tiles = valid_heads * head_tiles;
            const uint32_t first_tile = row * Wt + (head_start + first_head) * head_tiles;

            if (b != 0) {
                MaybeDeviceZoneScope("reader_reserve_x");
                x_dfb.reserve_back(block_tiles);
            }
            {
                MaybeDeviceZoneScope("reader_issue_x");
                uint32_t l1 = x_dfb.get_write_ptr();
                for (uint32_t i = 0; i < valid_tiles; ++i) {
                    noc_async_read_page(first_tile + i, x_acc, l1);
                    l1 += tile_bytes;
                    if ((i + 1) % read_chunk_tiles == 0) {
                        noc_async_read_barrier();
#ifdef COSSIN_TEAM
                        if (rope_pending) {
                            forward_rope();
                        }
#endif
                    }
                }
            }
            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
#ifdef COSSIN_TEAM
            if (b == 0) {
                if (rope_pending) {
                    forward_rope();
                }
                if (!is_leader) {
                    MaybeDeviceZoneScope("reader_rope_wait");
                    noc_semaphore_wait(valid_ptr, 1);
                    *valid_ptr = 0;
                }
            }
#endif
#ifdef HAS_ROPE
            if (b == 0) {
                cos_dfb.push_back(head_tiles);
                sin_dfb.push_back(head_tiles);
            }
#endif
            x_dfb.push_back(block_tiles);
        }
    }
#ifdef COSSIN_TEAM
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
