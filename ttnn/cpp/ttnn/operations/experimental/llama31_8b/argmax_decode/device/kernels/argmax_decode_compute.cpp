// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "argmax_decode_sfpu.hpp"
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_keys = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_keys);
    CircularBuffer in(cb_in);
    CircularBuffer keys(cb_keys);
    if (num_tiles == 0) {
        return;
    }
    in.wait_front(num_tiles);
    copy_init(cb_in);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, true>()));

    tile_regs_acquire();
    for (uint32_t t = 0; t < num_tiles; ++t) {
        copy_tile(cb_in, t, 0);
        MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, argmax_decode_step, (8), 0, VectorMode::RC, t, t == 0)));
    }
    tile_regs_commit();

    keys.reserve_back(1);
    tile_regs_wait();
    pack_tile(1, cb_keys);
    tile_regs_release();
    keys.push_back(1);
}
