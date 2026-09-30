// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre writer (BRISC, NoC1).
//
// load_w (once, before block 0; the writer is otherwise idle until the first partial): this rank's W slice ->
//   cb_weight, K order p = c*n + i (W page i*Ct + c_start + c), per the W role (RT arg; regime R2
//   `W column broadcast`, op_design.md Regimes):
//     W_ROLE_DRAM   — R1: the whole slice from DRAM, pushed in w_chunk_tiles chunks (the compute's fp32-W
//                     split of chunk j runs under the read of chunk j+1).
//     W_ROLE_SPREAD — column all-gather: every core of the physical column holds the same rank's slice; it
//                     reads only its share [own_p0, own_p1) from DRAM, (bf16 X / fp32 W) has its compute split
//                     that share in place (token CBs), multicasts it down the column (rotating-sender
//                     Mcast1D(PerColumn), Counter, write-once landing => no handshake), lands the other rows'
//                     w_events shares, and publishes the whole slice at once.
//   W rides the writer's NoC1 so it never queues behind the reader's X stream on NoC0.
// load_bias (once, after the W fill; only the coefficient phase needs it): the bias row (row 0 of faces 0/1:
//   the n*(n+2) <= 32 used columns, 2 x 64 B) -> cb_bias_coef in coefficient-major form (slot k, every lane =
//   b[k]; zero elsewhere).
//
// Per block, in this mandatory order (gather-slot reuse invariant):
//
//   send_partial_block        cb_partial [mix rows | sumsq rows] -> root cb_gathered slot [rank]
//                              (push model; slot layout [mix x block_token_tiles | sumsq x block_token_tiles]),
//                              then one semaphore increment on the root.
//   (root) gather credit       wait the monotonic arrival counter >= group_cores*(block+1), push cb_gathered.
//   broadcast_combined_block   root: SenderPipe::send of cb_combined over the group rectangle into every rank's
//                              cb_coef_in (its own copy by loopback); non-root: ReceiverPipe::receive into its
//                              cb_coef_in (the same address on every rank). The layout transforms to / from the
//                              coefficient-major form are the compute's (in DEST); the writer only moves tiles.
//   store_y_block              cb_y_out -> y DRAM, windows of y_chunk_tiles (wrap-aware on the CB ring).
//   store_post_comb_block      owned rows: cb_comb_coef [post, comb] row-major tiles -> DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "mhc_pre_layout.hpp"
#include "tools/profiler/kernel_profiler.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t MCAST_CT_BASE = 20;
constexpr uint32_t MCAST_RT_BASE = 17;
constexpr uint32_t W_ROLE_DRAM = 0;
constexpr uint32_t W_ROLE_SPREAD = 1;

void kernel_main() {
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered = get_compile_time_arg_val(1);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_comb_coef = get_compile_time_arg_val(4);
    constexpr uint32_t cb_y_out = get_compile_time_arg_val(5);
    constexpr uint32_t n_streams = get_compile_time_arg_val(6);
    constexpr uint32_t block_token_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t group_cores = get_compile_time_arg_val(9);
    constexpr uint32_t y_chunk_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t y_cb_pages = get_compile_time_arg_val(11);  // y_depth * y_chunk_tiles
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(12);
    constexpr uint32_t mix_cols = get_compile_time_arg_val(13);  // n*(n+2); slot mix_cols holds sum(x^2)
    constexpr uint32_t cb_bias_coef = get_compile_time_arg_val(14);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(15);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(16);   // R1 push quantum (split pipelining)
    constexpr uint32_t cb_w_own_ready = get_compile_time_arg_val(17);  // token: writer -> compute, share landed
    constexpr uint32_t cb_w_own_split = get_compile_time_arg_val(18);  // token: compute -> writer, share split
    constexpr bool w_presplit = get_compile_time_arg_val(19) != 0;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr auto w_mc = McastArgs<mc.next_compile_time_args_offset(), mc.next_runtime_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<w_mc.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_arg_val<uint32_t>(2);
    const uint32_t t_start = get_arg_val<uint32_t>(3);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(4);
    const uint32_t c_start = get_arg_val<uint32_t>(5);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(6);
    const uint32_t num_blocks = get_arg_val<uint32_t>(7);
    const uint32_t rank = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t b_addr = get_arg_val<uint32_t>(11);
    const uint32_t w_addr = get_arg_val<uint32_t>(12);
    const uint32_t w_role = get_arg_val<uint32_t>(13);
    const uint32_t own_p0 = get_arg_val<uint32_t>(14);  // W_ROLE_SPREAD: this core's share [own_p0, own_p1)
    const uint32_t own_p1 = get_arg_val<uint32_t>(15);
    const uint32_t w_events = get_arg_val<uint32_t>(16);  // shares multicast to this core by the other rows

    using mhc_layout::rc_index;
    using mhc_layout::slot_index;
    using mhc_layout::TILE_DATUMS;

    const uint32_t f_tile_bytes = get_tile_size(cb_partial);  // fp32 tile
    const uint32_t y_tile_bytes = get_tile_size(cb_y_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, f_tile_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, f_tile_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, f_tile_bytes);
    const uint32_t w_tile_bytes = get_tile_size(cb_weight);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    const uint32_t core_k_tiles = n_streams * core_c_tiles;

    Noc noc;

    CircularBuffer bias_cb(cb_bias_coef);
    cb_reserve_back(cb_bias_coef, 1);

    // ---- load_w ----
    auto read_w_tiles = [&](uint32_t w_base, uint32_t p0, uint32_t p1) {
        for (uint32_t p = p0; p < p1; ++p) {
            const uint32_t c = p / n_streams;
            const uint32_t i = p - c * n_streams;
            noc_async_read_page(i * tensor_c_tiles + c_start + c, w_acc, w_base + p * w_tile_bytes);
        }
    };
    auto load_bias = [&]() {
        // Row 0 of faces 0 and 1 (cols 0..15, 16..31) lands at its row-major offsets of the bias tile itself;
        // the values are picked up, then the tile is zeroed (unused slots / lanes must be 0) and filled.
        static_assert(mix_cols <= 32, "bias row must fit one tile row");
        constexpr uint32_t face_row_bytes = 16 * sizeof(float);
        const uint32_t bias_l1 = get_write_ptr(cb_bias_coef);
        noc_async_read(b_acc.get_noc_addr(0, rc_index(0, 0) * sizeof(float)), bias_l1, face_row_bytes);
        noc_async_read(
            b_acc.get_noc_addr(0, rc_index(0, 16) * sizeof(float)),
            bias_l1 + rc_index(0, 16) * sizeof(float),
            face_row_bytes);
        noc_async_read_barrier();
        float bvals[mix_cols];
        float* dst = reinterpret_cast<float*>(bias_l1);
        for (uint32_t k = 0; k < mix_cols; ++k) {
            bvals[k] = dst[rc_index(0, k)];
        }
        noc.async_write_zeros(bias_cb, f_tile_bytes);
        noc.write_zeros_l1_barrier();
        for (uint32_t k = 0; k < mix_cols; ++k) {
#pragma GCC unroll 8
            for (uint32_t l = 0; l < 32; ++l) {
                dst[slot_index(k, l)] = bvals[k];
            }
        }
        cb_push_back(cb_bias_coef, 1);
    };
    cb_reserve_back(cb_weight, core_k_tiles);
    const uint32_t w_base = get_write_ptr(cb_weight);
    if (w_role == W_ROLE_SPREAD) {
        {
            DeviceZoneScopedN("W-own");
            read_w_tiles(w_base, own_p0, own_p1);
            noc_async_read_barrier();
        }
        if constexpr (w_presplit) {
            cb_reserve_back(cb_w_own_ready, 1);
            cb_push_back(cb_w_own_ready, 1);
        }
        if constexpr (w_mc.active) {
            // Send only after the compute's in-place split finished (token); publish the slice only after the
            // send returned (source guard) and every other share landed (Counter): no L1 region is ever
            // written by two agents at once.
            if constexpr (w_presplit) {
                DeviceZoneScopedN("W-splitwait");
                cb_wait_front(cb_w_own_split, 1);
                cb_pop_front(cb_w_own_split, 1);
            }
            if (own_p1 > own_p0) {
                DeviceZoneScopedN("W-wsend");
                auto w_sender = w_mc.sender(noc);
                const uint32_t a = w_base + own_p0 * w_tile_bytes;
                w_sender.send(a, a, (own_p1 - own_p0) * w_tile_bytes);
            }
            DeviceZoneScopedN("W-wrecv");
            auto w_receiver = w_mc.receiver(noc);
            for (uint32_t e = 0; e < w_events; ++e) {
                w_receiver.receive();  // Counter: one event per other row's share, lands in place
            }
        }
        cb_push_back(cb_weight, core_k_tiles);
        // The bias is only needed at the first coefficients phase; its (tiny) DRAM read queues behind the X
        // burst at the bank, so it must not sit in front of the W exchange.
        {
            DeviceZoneScopedN("W-bias");
            load_bias();
        }
    } else {
        for (uint32_t p0 = 0; p0 < core_k_tiles; p0 += w_chunk_tiles) {
            const uint32_t p1 = (p0 + w_chunk_tiles) < core_k_tiles ? (p0 + w_chunk_tiles) : core_k_tiles;
            read_w_tiles(w_base, p0, p1);
            noc_async_read_barrier();
            cb_push_back(cb_weight, p1 - p0);
        }
        load_bias();
    }

    // Uniform-address CBs (allocated identically on every launched core).
    constexpr uint32_t slot_tiles = 2 * block_token_tiles;
    const uint32_t gathered_base = get_write_ptr(cb_gathered);
    const uint32_t sem_addr = get_semaphore(sem_gather_id);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t root_sem_noc = get_noc_addr(root_x, root_y, sem_addr);
    const uint64_t root_slot_noc = get_noc_addr(root_x, root_y, gathered_base + rank * slot_tiles * f_tile_bytes);

    // Group combine pipes (the group rectangle, root = rank 0 is the fixed sender, its own copy by loopback).
    auto sender = mc.sender(noc);
    auto receiver = mc.receiver(noc);

    uint32_t y_ring_pos = 0;  // position (in pages) of cb_y_out's read pointer within its ring

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;

        // ---- send_partial_block ----
        {
            DeviceZoneScopedN("W-pwait");
            cb_wait_front(cb_partial, 2 * extent);
        }
        {
            DeviceZoneScopedN("W-send");
            {
                // cb_partial holds [mix rows | sumsq rows] (the compute projects first); the slot keeps
                // [mix x block_token_tiles | sumsq x block_token_tiles] (one write when the block is full).
                const uint32_t src = get_read_ptr(cb_partial);
                if (extent == block_token_tiles) {
                    noc_async_write(src, root_slot_noc, 2 * extent * f_tile_bytes);
                } else {
                    noc_async_write(src, root_slot_noc, extent * f_tile_bytes);
                    noc_async_write(
                        src + extent * f_tile_bytes,
                        root_slot_noc + block_token_tiles * f_tile_bytes,
                        extent * f_tile_bytes);
                }
                noc_async_write_barrier();
                noc_semaphore_inc(root_sem_noc, 1);
            }
            cb_pop_front(cb_partial, 2 * extent);
        }

        // ---- gather credit (root) + broadcast_combined_block: S -> every rank's cb_coef_in ----
        // cb_coef_in advances identically on every rank of the group (same pushes / pops per block), so the
        // root's write pointer IS every receiver's landing address; the root's own copy rides the multicast
        // loopback (src cb_combined != dst cb_coef_in). Receivers reserve before they ack (PRE_HANDSHAKE).
        {
            DeviceZoneScopedN("W-gather");
            cb_reserve_back(cb_coef_in, slot_tiles);
            const uint32_t s_dst = get_write_ptr(cb_coef_in);
            if (rank == 0) {
                cb_reserve_back(cb_gathered, group_cores * slot_tiles);
                noc_semaphore_wait_min(sem_ptr, group_cores * (block_idx + 1));
                cb_push_back(cb_gathered, group_cores * slot_tiles);
                cb_wait_front(cb_combined, slot_tiles);
                const uint32_t s_src = get_read_ptr(cb_combined);
                if constexpr (group_cores > 1) {
                    sender.send(s_src, s_dst, slot_tiles * f_tile_bytes);
                } else {
                    noc_async_write(s_src, get_noc_addr(s_dst), slot_tiles * f_tile_bytes);
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_combined, slot_tiles);
            } else {
                receiver.receive();
            }
            cb_push_back(cb_coef_in, slot_tiles);
        }

        DeviceZoneScopedN("W-ystore");

        // ---- store_y_block ----
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t y_row_page = (t_start + row0 + t) * tensor_c_tiles + c_start;
            uint32_t c = 0;
            while (c < core_c_tiles) {
                uint32_t w = core_c_tiles - c;
                if (w > y_chunk_tiles) {
                    w = y_chunk_tiles;
                }
                if (w > y_cb_pages - y_ring_pos) {
                    w = y_cb_pages - y_ring_pos;
                }
                cb_wait_front(cb_y_out, w);
                const uint32_t rp = get_read_ptr(cb_y_out);
                for (uint32_t j = 0; j < w; ++j) {
                    noc_async_write_page(y_row_page + c + j, y_acc, rp + j * y_tile_bytes);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_y_out, w);
                y_ring_pos += w;
                if (y_ring_pos == y_cb_pages) {
                    y_ring_pos = 0;
                }
                c += w;
            }
        }

        // ---- store_post_comb_block (owned rows): cb_comb_coef [post, comb], already row-major tiles ----
        for (uint32_t t = 0; t < extent; ++t) {
            if (((row0 + t) % group_cores) != rank) {
                continue;
            }
            cb_wait_front(cb_comb_coef, 2);
            const uint32_t pc = get_read_ptr(cb_comb_coef);
            noc_async_write_page(t_start + row0 + t, post_acc, pc);
            noc_async_write_page(t_start + row0 + t, comb_acc, pc + f_tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_comb_coef, 2);
        }
    }
    noc_async_atomic_barrier();
}
