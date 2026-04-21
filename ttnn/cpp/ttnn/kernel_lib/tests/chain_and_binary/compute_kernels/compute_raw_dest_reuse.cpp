// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic kernel (experiment D): mirror DestReuseOp's per-tile LLK sequence
// using raw compute-api calls only, NO helper. If this reproduces the multi-tile
// hang observed in compute_binary_dest_reuse.cpp, the bug is in the LLK sequence
// itself (helper is innocent). If it works, the bug is in the helper's
// clashes_with_fpu reinit path orchestration.
//
// Sequence per tile (matches what DestReuseMul<cb_scale> + binary_op<SUB> emit):
//   cb_wait_front(cb_a, 1); cb_wait_front(cb_b, 1);
//   tile_regs_acquire();
//   sub_tiles_init(cb_a, cb_b);                      // AB unpack + ELWSUB math MOP
//   sub_tiles(cb_a, cb_b, 0, 0, 0);                  // DEST[0] = a - b
//   binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_scale);
//   binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_scale, 0, 0);
//   tile_regs_commit(); tile_regs_wait();
//   cb_reserve_back(cb_out, 1); pack_tile(0, cb_out); cb_push_back(cb_out, 1);
//   cb_pop_front(cb_a, 1); cb_pop_front(cb_b, 1);
//   tile_regs_release();

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_b = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_scale = static_cast<uint32_t>(tt::CBIndex::c_2);
    constexpr uint32_t cb_out = static_cast<uint32_t>(tt::CBIndex::c_16);

    binary_op_init_common(cb_a, cb_b, cb_out);

    cb_wait_front(cb_scale, 1);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // Match batch_norm's ordering: reserve output BEFORE acquire,
        // pop inputs AFTER release (not before).
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        sub_tiles_init(cb_a, cb_b);
        sub_tiles(cb_a, cb_b, 0, 0, 0);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_scale);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_scale, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }

    cb_pop_front(cb_scale, 1);
}
