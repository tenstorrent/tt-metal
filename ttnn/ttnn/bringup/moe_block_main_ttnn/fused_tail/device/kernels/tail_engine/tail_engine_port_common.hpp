// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the TailEngine gather ports (one per direction and link, in the column right of the engine). A port (d, l)
// sends half l of the K tiles of its blocks (own first, then the ones it relays) to the next chip in direction d,
// straight into the engine cores' landing slots there: share by share (row tile R of the block: its step there,
// column cx: K tiles kh(l) + ak(cx)), packets of up to RP tiles; a row tile's last packet of a group of WG columns
// (the watcher's) a fused write + increment of the watcher's landed word of that step - flushed behind the row tile's
// other packets on that link, so the watcher counts the step landed for its group's other cores. A share (own or
// relayed) is read from this chip's owner once landed.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t RT = get_compile_time_arg_val(0), GX = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t SHMAX = get_compile_time_arg_val(3), KT = get_compile_time_arg_val(4);
constexpr uint32_t PG = get_compile_time_arg_val(5);  // (packets a group: the reader's barrier, the sender's flush)
constexpr uint32_t PAIRS = get_compile_time_arg_val(6), WG = get_compile_time_arg_val(7);
// (the staging CB, the group counts CB, the poll scratch CB, and the engine's landing CB - also allocated here, at the
// same address as on the engine cores, so its base is the landing slots' address on every engine core)
constexpr uint32_t CBP = get_compile_time_arg_val(8), CBM = get_compile_time_arg_val(9);
constexpr uint32_t CB_POLL = get_compile_time_arg_val(10), CB_LAND = get_compile_time_arg_val(11);
// the engine's step words (this call's, CB_WD after the landing CB: one address on every engine and port core) and
// the zero gate's go a port kernel waits for before it reads or writes any of them (tail_engine_common.hpp)
constexpr uint32_t CB_WD = 16, PZR = 0, PZW = 1;
// (the go: the engine's lead increments it once a call once every engine core zeroed its words; re-armed here)
FORCE_INLINE void wait_zero_gate(uint32_t id) {
    volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
    noc_semaphore_wait_min(g, 1);
    noc_semaphore_inc(get_noc_addr(get_semaphore(id)), 0u - 1u);
    noc_async_atomic_barrier();
}

// the shares of a unit (block at chunk position pos on the chip that holds the slots), in the walk both kernels
// follow: f(pair, step, cx, k0, t0, tiles, last); part 0 every column, 1 / 2 the first / second half of them
template <typename F>
FORCE_INLINE void walk(uint32_t part, uint32_t pos, uint32_t ak, F&& f) {
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t t = pos * RT + r;
        for (uint32_t cx = 0; cx < GX; ++cx) {
            if ((part == 1 && cx >= GX / 2) || (part == 2 && cx < GX / 2)) {
                continue;
            }
            const uint32_t k0 = get_arg_val<uint32_t>(ak + 2 * cx), k1 = get_arg_val<uint32_t>(ak + 2 * cx + 1);
            for (uint32_t t0 = k0; t0 < k1; t0 += RP) {
                f(t % PAIRS, t / PAIRS, cx, k0, t0, (k1 - t0 < RP ? k1 - t0 : RP), t0 + RP >= k1);
            }
        }
    }
}
