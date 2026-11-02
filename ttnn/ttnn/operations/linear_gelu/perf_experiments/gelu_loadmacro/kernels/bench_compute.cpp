// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated pack-thread GELU bench: math copies each input tile into DEST (fp32), the pack thread applies
// the GELU variant `reps` times (as the op does, after the math->pack semaphore + dest-offset flip) and
// packs once. reps=0 measures the copy/pack/sync floor.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "gelu_variants.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_tile_init(cb_in);
    gv_init<variant>();

    cb_reserve_back(cb_in, num_tiles);
    cb_push_back(cb_in, num_tiles);
    cb_wait_front(cb_in, num_tiles);
    cb_reserve_back(cb_out, num_tiles);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_in, t, 0);
        tile_regs_commit();

        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t r = 0; r < reps; ++r) {
            gv_apply<variant>(0);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile(0, cb_out, t);
        tile_regs_release();
    }
    cb_push_back(cb_out, num_tiles);
    cb_pop_front(cb_in, num_tiles);
}
