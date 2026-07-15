// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "../zero_padded_kv_cache_common.hpp"

// Multiplies one (partial) tile by the row-mask for each of the Wt width-tiles, zeroing the pad rows
// while preserving the real rows. The reader and writer drive an UNCONDITIONAL src/mask/out CB
// protocol -- the reader always pushes src+mask and the writer always pops the out tiles (discarding
// the result on a chip with no partial pad work) -- so this kernel needs NO per-call scalar: it always
// processes exactly Wt tiles. Wt is a structural common arg (index 7) readable by all three compute
// threads, so no reader->compute handoff (control CB / mailbox) is required, and the kernel is
// identical on the scalar and metadata paths.
void kernel_main() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mask_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);

    const uint32_t Wt = get_common_arg_val<uint32_t>(7);  // structural; same on both paths

    CircularBuffer src(src_cb);
    CircularBuffer mask(mask_cb);
    CircularBuffer out(out_cb);

    binary_op_init_common(src_cb, mask_cb, out_cb);
    mul_tiles_init(src_cb, mask_cb);

    mask.wait_front(1);
    src.wait_front(Wt);
    out.reserve_back(Wt);
    for (uint32_t i = 0; i < Wt; ++i) {
        tile_regs_acquire();
        mul_tiles(src_cb, mask_cb, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    out.push_back(Wt);
    src.pop_front(Wt);
    mask.pop_front(1);
}
