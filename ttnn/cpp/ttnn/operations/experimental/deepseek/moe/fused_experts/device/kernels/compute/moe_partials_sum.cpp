// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/debug/kernel_profiler.hpp"

// persistent_moe_request's sum of one part of the row (see dataflow/moe_partials_sum.cpp): cb_in holds the op's
// chips' slots of that part, num_slots runs of tiles_per_part tiles; each output tile is the sum of that tile
// of every slot, taken in slot (chip) order, so every chip of the requester packs the same bits. The tiles are
// any whole-tile view of the bf16 part: the sum is elementwise.
//
// Compile-time args: 0: num_slots   1: tiles_per_part   2: cb_in   3: cb_out
// Runtime args: 0: skip (a stop request)

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    cb_wait_front(cb_in, num_slots * tiles_per_part);
    DeviceZoneScopedN("MOE_SUM");
    for (uint32_t t = 0; t < tiles_per_part; ++t) {
        tile_regs_acquire();
        copy_init(cb_in);
        copy_tile(cb_in, t, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in);
        for (uint32_t slot = 1; slot < num_slots; ++slot) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in, slot * tiles_per_part + t, 0);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_in, num_slots * tiles_per_part);
}
