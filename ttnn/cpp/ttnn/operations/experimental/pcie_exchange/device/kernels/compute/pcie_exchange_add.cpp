// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// out[t] = in[t] + landing[par][t] for the ntiles tiles of this core's shard. c_0 aliases the input shard,
// c_1 aliases both landing buffers (2 * ntiles pages), c_2 aliases the output shard. The parity of the landing
// buffer the peer wrote this time is the low bit of the local sequence number the dataflow kernel bumped.
//
// bfp8 payload mode: first pack the bf16 shard to bfp8 twice (c_3 for the dataflow kernel to ship, c_4 for the add),
// then out = bfp8(in) + bfp8(peer): both devices add the same two rounded operands, so the replicas stay identical.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage_dm = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stage_math = get_compile_time_arg_val(4);
    constexpr bool bfp8_payload = get_compile_time_arg_val(5) == 1;
    const uint32_t ntiles = get_arg_val<uint32_t>(0);
    const uint32_t ctrl = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_in, cb_landing, cb_out);

    uint32_t cb_a = cb_in;
    if constexpr (bfp8_payload) {
        copy_init(cb_in);
        pack_reconfig_data_format(cb_out, cb_stage_dm);
        cb_wait_front(cb_in, ntiles);
        for (uint32_t t = 0; t < ntiles; ++t) {
            cb_reserve_back(cb_stage_dm, 1);
            cb_reserve_back(cb_stage_math, 1);
            tile_regs_acquire();
            copy_tile(cb_in, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_stage_dm);
            pack_tile(0, cb_stage_math);
            tile_regs_release();
            cb_push_back(cb_stage_dm, 1);
            cb_push_back(cb_stage_math, 1);
        }
        cb_pop_front(cb_in, ntiles);
        pack_reconfig_data_format(cb_stage_dm, cb_out);
        reconfig_data_format(cb_stage_math, cb_landing);
        cb_a = cb_stage_math;
    }
    add_init(cb_a, cb_landing);

    cb_wait_front(cb_a, ntiles);
    cb_wait_front(cb_landing, 2 * ntiles);
    volatile uint32_t* local_seq = reinterpret_cast<volatile uint32_t*>(ctrl + 64);
    const uint32_t par = local_seq[0] & 1;

    for (uint32_t t = 0; t < ntiles; ++t) {
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        add_tiles(cb_a, cb_landing, t, par * ntiles + t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_a, ntiles);
    cb_pop_front(cb_landing, 2 * ntiles);
}
