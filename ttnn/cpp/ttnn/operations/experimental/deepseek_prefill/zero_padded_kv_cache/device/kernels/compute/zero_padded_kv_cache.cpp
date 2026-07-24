// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

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

    CircularBuffer src(src_cb);
    CircularBuffer mask(mask_cb);
    CircularBuffer out(out_cb);

    binary_op_init_common(src_cb, mask_cb, out_cb);
    mul_tiles_init(src_cb, mask_cb);

    mask.wait_front(1);
    src.wait_front(w.Wt);
    out.reserve_back(w.Wt);
    for (uint32_t i = 0; i < w.Wt; ++i) {
        tile_regs_acquire();
        mul_tiles(src_cb, mask_cb, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    out.push_back(w.Wt);
    src.pop_front(w.Wt);
    mask.pop_front(1);
}
