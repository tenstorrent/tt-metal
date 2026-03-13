// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-major element-wise binary compute kernel.
//
// Processes CB tiles produced by reader.cpp.  Each tile is a 1024-element
// hardware tile (CB page = STICK_SIZE * dtype_bytes = 2048/4096 bytes).
// The last tile may be partial (fewer valid elements), but the compute loop
// is identical — the writer is responsible for writing only valid bytes back.
//
// Required defines (set by host):
//   RM_BINARY_OP_ADD  — perform addition
//   RM_BINARY_OP_MUL  — perform multiplication
//   IS_FP32           — use fp32 SFPU path; omit for bf16 FPU path
//
// RT args:  [num_tiles]   (num_full_sticks + 1 if partial chunk exists)

#include <cstdint>

#include "api/compute/eltwise_binary.h"               // binary_op_init_common, add_tiles, mul_tiles
#include "api/compute/eltwise_binary_sfpu.h"          // add_binary_tile, mul_binary_tile, *_init
#include "api/compute/eltwise_unary/eltwise_unary.h"  // unary_op_init_common
#include "api/compute/tile_move_copy.h"               // copy_tile, copy_tile_to_dst_init_short_with_dt
#include "api/compute/eltwise_unary/fill.h"           // fill_tile, fill_tile_init

void kernel_main() {
    const uint32_t num_sticks = get_arg_val<uint32_t>(0);  // total tiles (full + partial)

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    binary_op_init_common(cb_a, cb_b, cb_out);

    DPRINT << "num_sticks: " << num_sticks << ENDL();

#ifdef IS_FP32
    BINARY_OP_INIT();
#else
    BINARY_OP_INIT(cb_a, cb_b);
#endif  // IS_FP32

    for (uint32_t i = 0; i < num_sticks; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

#ifdef IS_FP32
        // SFPU
        copy_tile_to_dst_init_short(cb_a);
        copy_tile(cb_a, 0, 0);

        copy_tile_to_dst_init_short(cb_b);
        copy_tile(cb_b, 0, 1);

        BINARY_OP(0, 1, 0);
#else
        // FPU
        BINARY_OP(cb_a, cb_b, 0, 0, 0);
#endif  // IS_FP32

        // fill_tile_init();
        // fill_tile(0, 3.f);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
