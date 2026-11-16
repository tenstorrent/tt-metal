// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/tensor_accessor.h"
#include "../persistent.h"

// The persistent op's sharded router, the reducer core (RISCV_0). This chip scores experts
// [32 * my_chip, +32); per iteration this kernel
//   1. reads the bias row into cb_rbias (bias_tiles 1x32 tiles; this chip's experts' are tile my_chip);
//   2. once the partial cores have written their partials into cb_parts (the parts count reaches num_parts per
//      iteration), publishes them for router_reduce.cpp;
//   3. takes its bf16 [score, ranking] tiles from cb_rout and keeps the top_k of its 32 experts by ranking, best
//      first and of equal rankings the lower id first, as (order key, global expert id, fp32 score) triplets --
//      the candidates router_topk.cpp makes for one tile column;
//   4. writes them into this chip's slot of the row's requester's candidate slots (the header's kFeHdrRequester:
//      one set per requester, each with one row in flight), at the same address on the route merge core (whose
//      sem_router it bumps) and on the exchange sender, which forwards them to every other chip's route merge
//      (fe_reduce_sender.cpp). A requester's slot is written again only for its next row, which it sends once every
//      chip's partials of this one are back -- after every chip has merged these candidates, so after the sender
//      has forwarded them.
//
// Compile-time args:
//   0: cb_parts   1: num_parts   2: cb_rbias   3: cb_rout   4: top_k   5: my_chip   6: cand_slot_bytes
//   7: num_slots (chips)   8: sem_router (on the route merge)   9: bias_tiles   10: bias row bytes
//   11+: TensorAccessorArgs(bias), then rows (the request's token rows, the height of the score tiles)
//
// Several rows: cb_rout's tiles are (rows x 32), expert i of row r at face i / 16, face row r; a slot holds each row's
// top_k in turn (row r's at triplet r * top_k).
//
// FE_NATIVE_ONE_ROW: a one-row request (persistent.h kFeHdrRows) runs on the CBs' one-row aliases -- the bias row as
// bias_tiles 1x32 tiles (as a one-row op's), the score tiles 1x32 -- and its one row's top-k is this RISC's alone.
//
// Runtime args:
//   0: candidate slots address   1: route merge NoC x   2: route merge NoC y   3: sender NoC x   4: sender NoC y
//   5: parts count address   6: the sender's ready count address   7: forwarded count address (on this core)

namespace {
// Monotonic in the fp32 value: bigger value <=> bigger key (the route merge merges on these).
FORCE_INLINE uint32_t f32_order_key(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_parts_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_parts = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rbias_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rout_id = get_compile_time_arg_val(3);
    constexpr uint32_t top_k = get_compile_time_arg_val(4);
    constexpr uint32_t my_chip = get_compile_time_arg_val(5);
    constexpr uint32_t cand_slot_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_slots = get_compile_time_arg_val(7);
    constexpr uint32_t sem_router_id = get_compile_time_arg_val(8);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t bias_row_bytes = get_compile_time_arg_val(10);
    constexpr auto bias_args = TensorAccessorArgs<11>();
    constexpr uint32_t rows = get_compile_time_arg_val(bias_args.next_compile_time_args_offset());

    const uint32_t cand_addr = get_arg_val<uint32_t>(0);
    const uint32_t merge_x = get_arg_val<uint32_t>(1);
    const uint32_t merge_y = get_arg_val<uint32_t>(2);
    const uint32_t sender_x = get_arg_val<uint32_t>(3);
    const uint32_t sender_y = get_arg_val<uint32_t>(4);
    auto* parts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
    const uint32_t sender_ready_addr = get_arg_val<uint32_t>(6);
    auto* forwarded = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(7));

    // Rows [r0, r1) of the [score, ranking] tiles at words_l1 (two bf16 per word: expert i of row r at face i / 16,
    // face row r), each row's top_k into the slot at slot_l1, best first, ties to the lower expert. The tiles are
    // (tile_rows x 32): `rows`, or 1 for a native one-row request.
    auto topk_rows = [&](uint32_t words_l1, uint32_t slot_l1, uint32_t r0, uint32_t r1, uint32_t tile_rows) {
        // A (tile_rows x 32) bf16 tile: two faces of tile_rows x 16, so a tile is 32 * tile_rows half-words.
        const uint32_t tile_halves = 32u * tile_rows;
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_l1);
        auto* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1);
        for (uint32_t r = r0; r < r1; ++r) {
            uint32_t key[32];
            for (uint32_t f = 0; f < 2u; ++f) {
                for (uint32_t w = 0; w < 8u; ++w) {
                    const uint32_t pair = words[tile_halves / 2u + (f * tile_rows + r) * 8u + w];
                    key[f * 16u + 2u * w] = f32_order_key(pair << 16);
                    key[f * 16u + 2u * w + 1u] = f32_order_key(pair & 0xFFFF0000u);
                }
            }
            // One pass keeping the best top_k sorted: an expert goes after every held one of an equal or bigger key,
            // so equal keys keep the lower expert first -- what picking the first maximum top_k times gives.
            uint32_t best_key[top_k];
            uint32_t best_at[top_k];
            uint32_t held = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint32_t k = key[i];
                if (held == top_k && k <= best_key[top_k - 1u]) {
                    continue;
                }
                uint32_t p = held < top_k ? held++ : top_k - 1u;
                while (p > 0u && best_key[p - 1u] < k) {
                    best_key[p] = best_key[p - 1u];
                    best_at[p] = best_at[p - 1u];
                    --p;
                }
                best_key[p] = k;
                best_at[p] = i;
            }
            for (uint32_t j = 0; j < top_k; ++j) {
                const uint32_t best = best_at[j];
                const uint32_t pair = words[((best >> 4) * tile_rows + r) * 8u + ((best & 15u) >> 1)];
                const uint32_t at = 3u * (r * top_k + j);
                cand[at] = best_key[j];
                cand[at + 1u] = my_chip * 32u + best;
                cand[at + 2u] = (best & 1u) != 0u ? (pair & 0xFFFF0000u) : (pair << 16);
            }
        }
    };

#if defined(FE_TOPK_HELPER)
    // The reducer's other RISC: the second half of the rows' top-k, each several-row iteration once the reader has
    // published the tiles' address (FE_TOPK_SEM_ADDR) and the count of such iterations (FE_TOPK_SEM_GO: a native
    // one-row iteration is the reader's alone, so it counts none); its slot is at the same address.
    (void)merge_x;
    (void)merge_y;
    (void)sender_x;
    (void)sender_y;
    (void)parts;
    (void)sender_ready_addr;
    (void)forwarded;
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_GO));
    for (uint32_t it = 0;; ++it) {
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *go;
        } while ((v & ~kFeStopBit) < it + 1u);
        if ((v & kFeStopBit) != 0u) {
            break;
        }
        const uint32_t words_l1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_ADDR));
        topk_rows(
            words_l1,
            cand_addr + (fe_header(kFeHdrRequester) * num_slots + my_chip) * cand_slot_bytes,
            rows / 2u,
            rows,
            rows);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_DONE)) = it + 1u;
    }
    return;
#endif

    uint32_t iter = 0;
    // The several-row iterations handed to the helper (FE_TOPK_SEM_GO counts these).
    uint32_t helper_iters = 0;
    (void)helper_iters;
    for (; fe_begin_iteration(iter, cb_parts_id, num_parts, /*has_writer=*/false); ++iter) {
        const bool one_row = fe_one_row();
        const uint32_t cb_rbias = fe_cb(cb_rbias_id, one_row);
        const uint32_t cb_parts = fe_cb(cb_parts_id, one_row);
        const uint32_t cb_rout = fe_cb(cb_rout_id, one_row);
        const uint32_t tile_rows = one_row ? 1u : rows;
        // 1. The bias does not depend on the row.
        cb_reserve_back(cb_rbias, bias_tiles);
        const auto bias = TensorAccessor(bias_args, fe_header(kFeHdrRouterBias));
        if (rows == 1u || one_row) {
            // The row is bias_tiles 1x32 tiles: tile my_chip is this chip's experts' biases.
            noc_async_read(bias.get_noc_addr(0), get_write_ptr(cb_rbias), bias_row_bytes);
            noc_async_read_barrier();
        } else {
            // Taller tiles: tile 0 is this chip's 32 biases in every row (router_reduce.cpp's compute takes tile 0),
            // built from the row read into the CB's last bias_row_bytes.
            constexpr uint32_t tile_bytes = 64u * rows;
            constexpr uint32_t raw_offset = (bias_tiles * tile_bytes - bias_row_bytes) & ~63u;
            static_assert(rows == 1u || raw_offset >= tile_bytes, "the bias row must land past tile 0");
            const uint32_t base = get_write_ptr(cb_rbias);
            noc_async_read(bias.get_noc_addr(0), base + raw_offset, bias_row_bytes);
            noc_async_read_barrier();
            auto* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + raw_offset) + my_chip * 16u;
            auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for (uint32_t f = 0; f < 2u; ++f) {
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t w = 0; w < 8u; ++w) {
                        tile[(f * rows + r) * 8u + w] = raw[f * 8u + w];
                    }
                }
            }
        }
        cb_push_back(cb_rbias, bias_tiles);

        // 2.
        {
            DeviceZoneScopedN("FE_R_PARTS_WAIT");
            noc_semaphore_wait_min(parts, num_parts * (iter + 1u));
        }
        cb_reserve_back(cb_parts, num_parts);
        cb_push_back(cb_parts, num_parts);

        // 3. A 1x32 tile is its 32 values in column order: expert i's score is half-word i, its ranking 32 + i;
        // row r of a taller tile is face row r of both faces.
        cb_wait_front(cb_rout, 2);
        invalidate_l1_cache();
        const uint32_t slot_l1 = cand_addr + (fe_header(kFeHdrRequester) * num_slots + my_chip) * cand_slot_bytes;
        {
            DeviceZoneScopedN("FE_R_TOPK");
            // Both tiles' rows, two bf16 per word: score (tile 0) and ranking (tile 1) of expert i of row r at face
            // i / 16, face row r -- read once as words into registers.
            const uint32_t words_l1 = get_read_ptr(cb_rout);
#if defined(FE_TOPK_SEM_GO)
            // Several rows: the reducer's other RISC (FE_TOPK_HELPER) takes the second half of them.
            if (rows > 1u && !one_row) {
                ++helper_iters;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_ADDR)) = words_l1;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_GO)) = helper_iters;
                topk_rows(words_l1, slot_l1, 0u, rows / 2u, rows);
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_DONE)), helper_iters);
            } else {
                topk_rows(words_l1, slot_l1, 0u, tile_rows, tile_rows);
            }
#else
            topk_rows(words_l1, slot_l1, 0u, tile_rows, tile_rows);
#endif
        }
        cb_pop_front(cb_rout, 2);

        // 4.
        noc_async_write(slot_l1, get_noc_addr(merge_x, merge_y, slot_l1), cand_slot_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(sem_router_id)), 1);
        noc_async_write(slot_l1, get_noc_addr(sender_x, sender_y, slot_l1), cand_slot_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, sender_ready_addr), 1);
        noc_async_atomic_barrier();
    }
    // Every partial and every forward of this launch has landed once the sender has forwarded them all; the counts
    // start at zero next launch.
    noc_semaphore_wait_min(forwarded, iter);
    *parts = 0;
    *forwarded = 0;
#if defined(FE_TOPK_SEM_GO)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_TOPK_SEM_GO)) = kFeStopBit | (helper_iters + 1u);
#endif
}
