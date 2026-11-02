// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "tools/profiler/kernel_profiler.hpp"

// Router reader + partial top-k kernel (RISCV_0 of each router core, the row below the expert grid).
//
// This core's gate tile columns are slot, slot + col_stride, ... (n_tiles of them), each one [D, 32]
// ND shard of k_tiles tiles in K order. The kernel
//   1. publishes the activation (cb_rin is aliased over this core's own replica shard), reads the whole
//      bf16 bias row into cb_rbias, and fills cb_rw with the weight: an L1-resident gate is already
//      there (cb_rw is aliased over this core's shards, back to back), a DRAM gate is streamed in
//      piece_tiles-tile K pieces, column by column, with two pieces in flight;
//   2. takes router_scores.cpp's bf16 [scores | ranking] tiles from cb_rout and keeps the top_k of its
//      n_tiles * 32 experts by ranking, best first and of equal rankings the lower id first;
//   3. writes them to slot `slot` of the leader's cb_routing as (order key, expert id, fp32 score)
//      triplets, staged at the same address in its own cb_routing, and bumps the leader's sem_router.
//
// Compile-time args:
//   0: cb_rin          1: k_tiles         2: cb_rw
//   3: cb_rbias        4: bias_page_bytes (the bias row)      5: bias_tiles (E / 32)
//   6: cb_rout         7: cb_routing      8: cand_offset      9: cand_slot_bytes
//   10: sem_router     11: top_k          12: col_stride (the router core count)
//   13: gate_in_dram   14: piece_tiles    15: gate tile bytes
//   16+: TensorAccessorArgs(bias), then TensorAccessorArgs(gate)
//
// Runtime args:
//   0: bias address   1: gate address   2: n_tiles   3: slot (this core's index, also its first tile column)
//   4: leader_noc_x   5: leader_noc_y

namespace {

// Monotonic in the fp32 value: bigger value <=> bigger key (the leader merges on these).
FORCE_INLINE uint32_t f32_order_key(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rin_id = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rw_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rbias_id = get_compile_time_arg_val(3);
    constexpr uint32_t bias_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rout_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_routing_id = get_compile_time_arg_val(7);
    constexpr uint32_t cand_offset = get_compile_time_arg_val(8);
    constexpr uint32_t cand_slot_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t sem_router_id = get_compile_time_arg_val(10);
    constexpr uint32_t top_k = get_compile_time_arg_val(11);
    constexpr uint32_t col_stride = get_compile_time_arg_val(12);
    constexpr bool gate_in_dram = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t piece_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t gate_tile_bytes = get_compile_time_arg_val(15);
    constexpr auto bias_args = TensorAccessorArgs<16>();
    constexpr auto gate_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();

    const uint32_t bias_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t slot = get_arg_val<uint32_t>(3);
    const uint32_t leader_noc_x = get_arg_val<uint32_t>(4);
    const uint32_t leader_noc_y = get_arg_val<uint32_t>(5);

    Noc noc;
    CircularBuffer cb_rin(cb_rin_id);
    CircularBuffer cb_rw(cb_rw_id);
    CircularBuffer cb_rbias(cb_rbias_id);
    CircularBuffer cb_rout(cb_rout_id);

    cb_rin.reserve_back(k_tiles);
    cb_rin.push_back(k_tiles);

    cb_rbias.reserve_back(bias_tiles);
    noc.async_read(
        TensorAccessor(bias_args, bias_addr), cb_rbias, bias_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_rbias.push_back(bias_tiles);

    cb_rw.reserve_back(n_tiles * k_tiles);
    if constexpr (gate_in_dram) {
        // cb_rw holds all n_tiles columns unwrapped, so piece q lands at base + q * piece_bytes. Piece q + 1
        // is issued before piece q is waited on (by its own transaction id), so the reads stay back to back
        // while each landed piece goes to compute.
        constexpr uint32_t pieces_per_col = k_tiles / piece_tiles;
        constexpr uint32_t piece_bytes = piece_tiles * gate_tile_bytes;
        constexpr uint32_t kPieceTrid[2] = {1u, 2u};
        const auto gate_accessor = TensorAccessor(gate_args, gate_addr);
        ShardView gate(gate_accessor);
        const uint32_t base = cb_rw.get_write_ptr();
        const uint32_t pieces = n_tiles * pieces_per_col;
        auto issue_piece = [&](uint32_t q) {
            noc.async_read<NocOptions::TXN_ID>(
                gate,
                CoreLocalMem<uint32_t>(base + q * piece_bytes),
                piece_bytes,
                {.shard_id = slot + (q / pieces_per_col) * col_stride,
                 .offset_bytes = (q % pieces_per_col) * piece_bytes},
                {.offset_bytes = 0},
                {.trid = kPieceTrid[q % 2u]});
        };
        issue_piece(0);
        for (uint32_t q = 0; q < pieces; ++q) {
            if (q + 1u < pieces) {
                issue_piece(q + 1u);
            }
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = kPieceTrid[q % 2u]});
            cb_rw.push_back(piece_tiles);
        }
        noc_async_read_set_trid(0, noc.get_noc_id());
    } else {
        cb_rw.push_back(n_tiles * k_tiles);
    }

    // cb_rout: n_tiles bf16 1x32 score tiles, then their n_tiles ranking tiles. A 1x32 tile is its 32
    // values in column order, so local expert i's score is half-word i and its ranking n_tiles*32 + i.
    // A bf16 is the top half of the fp32 it widens to.
    cb_rout.wait_front(2u * n_tiles);
    CoreLocalMem<volatile uint16_t> vals(cb_rout.get_read_ptr());
    const uint32_t slot_l1 = CircularBuffer(cb_routing_id).get_write_ptr() + cand_offset + slot * cand_slot_bytes;
    {
        DeviceZoneScopedN("FE_R_TOPK");
        // The same sorted running list as the leader's ranking scan, on fp32 order keys: a value enters
        // only if it beats the k-th best (strict >), after any equal ones already held. Local experts
        // run in ascending global id (column slot + n * col_stride), so equal values keep the lower id.
        uint32_t best_key[top_k];
        uint32_t best_idx[top_k];
        const uint32_t n_experts = n_tiles * 32u;
        for (uint32_t i = 0; i < n_experts; ++i) {
            const uint32_t key = f32_order_key(static_cast<uint32_t>(vals[n_experts + i]) << 16);
            if (i < top_k || key > best_key[top_k - 1u]) {
                uint32_t p = 0;
                while (p < i && p + 1u < top_k && best_key[p] >= key) {
                    ++p;
                }
                for (uint32_t j = (i < top_k ? i : top_k - 1u); j > p; --j) {
                    best_key[j] = best_key[j - 1u];
                    best_idx[j] = best_idx[j - 1u];
                }
                best_key[p] = key;
                best_idx[p] = i;
            }
        }
        CoreLocalMem<volatile uint32_t> cand(slot_l1);
        for (uint32_t j = 0; j < top_k; ++j) {
            const uint32_t i = best_idx[j];
            cand[3u * j] = best_key[j];
            cand[3u * j + 1u] = (slot + (i >> 5) * col_stride) * 32u + (i & 31u);
            cand[3u * j + 2u] = static_cast<uint32_t>(vals[i]) << 16;
        }
    }
    cb_rout.pop_front(2u * n_tiles);

    noc.async_write(
        CoreLocalMem<uint32_t>(slot_l1),
        UnicastEndpoint{},
        cand_slot_bytes,
        {.offset_bytes = 0},
        {.noc_x = leader_noc_x, .noc_y = leader_noc_y, .addr = slot_l1});
    noc.async_write_barrier();
    Semaphore<>(sem_router_id).up(noc, leader_noc_x, leader_noc_y, 1);
    // Semaphore::up is a non-posted atomic; async_write_barrier does not drain it.
    noc.async_atomic_barrier();
}
