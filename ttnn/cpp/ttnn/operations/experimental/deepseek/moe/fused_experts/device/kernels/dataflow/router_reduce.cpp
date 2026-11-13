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
//   4. writes them into this chip's slot of the iteration's half of the candidate slots, at the same address on
//      the route merge core (whose sem_router it bumps) and on the exchange sender, which forwards them to every
//      other chip's route merge (fe_reduce_sender.cpp). The sender's half is reused two iterations on, so before
//      writing iteration i there it waits until the sender has forwarded iteration i - 2.
//
// Compile-time args:
//   0: cb_parts   1: num_parts   2: cb_rbias   3: cb_rout   4: top_k   5: my_chip   6: cand_slot_bytes
//   7: num_slots (chips)   8: sem_router (on the route merge)   9: bias_tiles   10: bias row bytes
//   11+: TensorAccessorArgs(bias)
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

    const uint32_t cand_addr = get_arg_val<uint32_t>(0);
    const uint32_t merge_x = get_arg_val<uint32_t>(1);
    const uint32_t merge_y = get_arg_val<uint32_t>(2);
    const uint32_t sender_x = get_arg_val<uint32_t>(3);
    const uint32_t sender_y = get_arg_val<uint32_t>(4);
    auto* parts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
    const uint32_t sender_ready_addr = get_arg_val<uint32_t>(6);
    auto* forwarded = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(7));

    uint32_t iter = 0;
    for (; fe_begin_iteration(iter, cb_parts_id, num_parts, /*has_writer=*/false); ++iter) {
        // 1. The bias does not depend on the row.
        cb_reserve_back(cb_rbias_id, bias_tiles);
        const auto bias = TensorAccessor(bias_args, fe_header(kFeHdrRouterBias));
        noc_async_read(bias.get_noc_addr(0), get_write_ptr(cb_rbias_id), bias_row_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_rbias_id, bias_tiles);

        // 2.
        {
            DeviceZoneScopedN("FE_R_PARTS_WAIT");
            noc_semaphore_wait_min(parts, num_parts * (iter + 1u));
        }
        cb_reserve_back(cb_parts_id, num_parts);
        cb_push_back(cb_parts_id, num_parts);

        // 3. A 1x32 tile is its 32 values in column order: expert i's score is half-word i, its ranking 32 + i.
        cb_wait_front(cb_rout_id, 2);
        const uint32_t slot_l1 = cand_addr + ((iter & 1u) * num_slots + my_chip) * cand_slot_bytes;
        {
            DeviceZoneScopedN("FE_R_TOPK");
            auto* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_rout_id));
            uint32_t best_key[top_k];
            uint32_t best_idx[top_k];
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint32_t key = f32_order_key(static_cast<uint32_t>(vals[32u + i]) << 16);
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
            auto* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1);
            for (uint32_t j = 0; j < top_k; ++j) {
                const uint32_t i = best_idx[j];
                cand[3u * j] = best_key[j];
                cand[3u * j + 1u] = my_chip * 32u + i;
                cand[3u * j + 2u] = static_cast<uint32_t>(vals[i]) << 16;
            }
        }
        cb_pop_front(cb_rout_id, 2);

        // 4.
        noc_async_write(slot_l1, get_noc_addr(merge_x, merge_y, slot_l1), cand_slot_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(sem_router_id)), 1);
        if (iter >= 2u) {
            DeviceZoneScopedN("FE_R_FWD_WAIT");
            noc_semaphore_wait_min(forwarded, iter - 1u);
        }
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
}
