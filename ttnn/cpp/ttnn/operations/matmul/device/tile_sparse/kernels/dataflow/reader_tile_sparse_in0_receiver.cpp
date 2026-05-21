// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-sparse matmul: in0 (A matrix) mcast receiver kernel with K-block skip.
//
// Paired with reader_tile_sparse_in0_sender.cpp. For each K-block, checks the
// global k_active_mask bitmask:
//   - Active  (bit=1): participates in mcast protocol (signals sender, waits for
//                      data, pushes to CB).
//   - Inactive(bit=0): skips the K-block entirely — no CB ops, no semaphores.
//
// Because sender and receiver check the SAME bitmask, semaphore counts remain
// consistent and no CB data is produced for inactive K-blocks.
//
// Compile-time args:
//   0  in0_block_num_tiles  -- tiles per K-block (in0_block_h * in0_block_w)
//   1  num_k_blocks         -- total K-blocks = Kt / in0_block_w
//   2  num_m_blocks         -- M outer blocks = per_core_M / out_block_h
//   3  sender_semaphore_id
//   4  receiver_semaphore_id
//
// Named compile-time arg: cb_in0
//
// Runtime args:
//   0  sender_noc_x   -- physical NOC X of sender (core 0)
//   1  sender_noc_y   -- physical NOC Y of sender (core 0)
//   2  k_active_mask  -- bitmask of active K-blocks (must match sender's bitmask)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // ---- Runtime args ----
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t k_mask = get_arg_val<uint32_t>(2);

    // ---- Compile-time args ----
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(4);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");

    const uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_sem_addr);

    for (uint32_t bm = 0; bm < num_m_blocks; ++bm) {
        for (uint32_t bk = 0; bk < num_k_blocks; ++bk) {
            if (!((k_mask >> bk) & 1u)) {
                continue;  // Skip inactive K-block
            }

            cb_reserve_back(cb_in0, in0_block_num_tiles);

            // Signal sender we are ready for this block
            noc_semaphore_set(receiver_sem_ptr, INVALID);
            noc_semaphore_inc(sender_sem_noc_addr, 1);

            // Wait for sender to multicast the data
            noc_semaphore_wait(receiver_sem_ptr, VALID);

            cb_push_back(cb_in0, in0_block_num_tiles);
        }
    }
}
