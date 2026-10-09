// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-matmul receiver data movement (BRISC) on a compute core. Per expert: publishes the resident in0 (x,
// pre-blocked by K-block in its CB) to compute, then runs the in1 landing ring: grants the forwarder one credit per
// free slot (non-blocking) and publishes each block once the forwarder's data counter says it has landed; finally
// drains the expert's output (out CB backed by the output shard, which keeps the last expert's result).
//
// CT: 0 IN0_CB, 1 IN0_TILES, 2 IN1_CB, 3 BLK_TILES, 4 BLOCKS_PER_EXPERT, 5 NUM_EXPERTS, 6 SLOTS, 7 OUT_CB,
//     8 OUT_TILES, 9 DATA_SEM
// RT: 0 forwarder packed (x << 16 | y), 1 credit semaphore id on the forwarder
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t blk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_expert = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t slots = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t total_blocks = blocks_per_expert * num_experts;

    const uint32_t fxy = get_arg_val<uint32_t>(0);
    const uint32_t credit_sem_id = get_arg_val<uint32_t>(1);
    const uint64_t credit_noc = get_noc_addr(fxy >> 16, fxy & 0xFFFF, get_semaphore(credit_sem_id));
    volatile tt_l1_ptr uint32_t* data_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_sem_id));

    // Every slot starts free.
    noc_semaphore_inc(credit_noc, slots);
    uint32_t granted = slots;
    uint32_t pushed = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        cb_reserve_back(in0_cb, in0_tiles);  // compute done with the previous expert's in0 (same data, same memory)
        cb_push_back(in0_cb, in0_tiles);
        const uint32_t end = pushed + blocks_per_expert;
        while (pushed < end) {
            // Grant a credit for the next slot as soon as compute has freed it.
            if (granted < total_blocks && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * blk_tiles)) {
                noc_semaphore_inc(credit_noc, 1);
                ++granted;
                continue;
            }
            invalidate_l1_cache();
            if (*data_sem > pushed) {
                cb_push_back(in1_cb, blk_tiles);
                ++pushed;
            }
        }
        cb_wait_front(out_cb, out_tiles);
        if (e + 1 < num_experts) {
            cb_pop_front(out_cb, out_tiles);
        }
    }
    noc_async_atomic_barrier();
}
