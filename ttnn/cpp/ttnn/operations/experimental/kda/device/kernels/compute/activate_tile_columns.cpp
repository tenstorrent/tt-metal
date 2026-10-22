// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"

// The BF16-grade sigmoid of this core's tiles, `group` per DST acquire.
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    const uint32_t tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    sigmoid_tile_init();
    for (uint32_t first = 0; first < tiles; first += group) {
        const uint32_t count = tiles - first < group ? tiles - first : group;
        cb_wait_front(cb_in, count);
        cb_reserve_back(cb_out, count);
        tile_regs_acquire();
        for (uint32_t i = 0; i < count; ++i) {
            copy_tile(cb_in, i, i);
            sigmoid_tile<VectorMode::RC, false, false>(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < count; ++i) {
            pack_tile(i, cb_out, i);
        }
        tile_regs_release();
        cb_push_back(cb_out, count);
        cb_pop_front(cb_in, count);
    }
}
