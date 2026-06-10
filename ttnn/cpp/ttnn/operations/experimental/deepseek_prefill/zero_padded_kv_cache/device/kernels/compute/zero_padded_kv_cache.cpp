// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "../zero_padded_kv_cache_common.hpp"

// Multiplies the boundary (partial) pad tile by the row-mask, zeroing the pad rows while preserving
// the real rows. Only runs on the chip that owns the partial tile. The mask is a single tile reused
// across all Wt width-tiles (the mask depends only on the row).
void kernel_main() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mask_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);

    const ZeroPadChipWork w = zero_pad_compute_chip_work();
    if (w.count == 0 || w.first_partial == 0) {
        return;
    }

    binary_op_init_common(src_cb, mask_cb, out_cb);
    mul_tiles_init(src_cb, mask_cb);

    cb_wait_front(mask_cb, 1);
    cb_wait_front(src_cb, w.Wt);
    cb_reserve_back(out_cb, w.Wt);
    for (uint32_t i = 0; i < w.Wt; ++i) {
        tile_regs_acquire();
        mul_tiles(src_cb, mask_cb, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, w.Wt);
    cb_pop_front(src_cb, w.Wt);
    cb_pop_front(mask_cb, 1);
}
