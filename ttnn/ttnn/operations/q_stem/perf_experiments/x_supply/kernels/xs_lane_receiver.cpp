// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench, CANDIDATE receiver for xs_lane_sender.cpp lanes (raw; bypasses the Mcast helper receiver).
// Same credit loop as the op's receive_x_rounds (credit slot a to every lane that sends round a as soon as the slot
// is free, up to `ahead` rounds ahead), but every (lane, slot) has its own data-ready flag, so `ahead` may be the
// full ring depth: a flag word is reused only after the receiver cleared it and freed its slot.
// RT: n_lanes, then per lane: credit_x, credit_y (virtual), sem_credit, sem_flag0, r0, rstep.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t n_rounds = get_compile_time_arg_val(2);
    constexpr uint32_t depth = get_compile_time_arg_val(3);
    constexpr uint32_t ahead = get_compile_time_arg_val(4);
    constexpr uint32_t MAXL = 4;

    const uint32_t n_lanes = get_arg_val<uint32_t>(0);
    uint64_t credit_addr[MAXL];
    uint32_t flag0[MAXL], lr0[MAXL], lstep[MAXL];
    for (uint32_t l = 0; l < n_lanes; ++l) {
        const uint32_t b = 1 + 6 * l;
        credit_addr[l] = get_noc_addr(get_arg_val<uint32_t>(b), get_arg_val<uint32_t>(b + 1),
                                      get_semaphore(get_arg_val<uint32_t>(b + 2)));
        flag0[l] = get_semaphore(get_arg_val<uint32_t>(b + 3));
        lr0[l] = get_arg_val<uint32_t>(b + 4);
        lstep[l] = get_arg_val<uint32_t>(b + 5);
    }
    // semaphores are 16 B apart (get_semaphore(id) = base + id * L1_ALIGNMENT)
    const uint32_t sem_stride = get_semaphore(1) - get_semaphore(0);
    auto covers = [&](uint32_t l, uint32_t r) { return r >= lr0[l] && (r - lr0[l]) % lstep[l] == 0; };

    uint32_t acked = 0;
    auto credit = [&](uint32_t r) {
        while (acked < n_rounds && acked < r + ahead) {
            const uint32_t pages = (acked - r + 1) * round_tiles;
            if (acked == r) {
                while (!cb_pages_reservable_at_back(cb_x, pages)) {
                }
            } else if (!cb_pages_reservable_at_back(cb_x, pages)) {
                break;
            }
            for (uint32_t l = 0; l < n_lanes; ++l) {
                if (covers(l, acked)) {
                    noc_semaphore_inc(credit_addr[l], 1);
                }
            }
            ++acked;
        }
    };
    for (uint32_t r = 0; r < n_rounds; ++r) {
        credit(r);
        [[maybe_unused]] const uint32_t slot = r % depth;
        for (uint32_t l = 0; l < n_lanes; ++l) {
            if (covers(l, r)) {
#ifdef XS_SETCOUNT
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag0[l]), (r - lr0[l]) / lstep[l] + 1);
#else
                volatile tt_l1_ptr uint32_t* f =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag0[l] + slot * sem_stride);
                noc_semaphore_wait(f, 1);
                *f = 0;
#endif
            }
        }
        cb_push_back(cb_x, round_tiles);
    }
    noc_async_atomic_barrier();
}
