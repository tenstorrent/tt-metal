// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    const bool scalar_is_not_1 = scalar_arg != 1u;

    CircularBuffer cb_in0(tt::CBIndex::c_0);  // input_a
    CircularBuffer cb_in1(tt::CBIndex::c_1);  // input_b
    CircularBuffer cb_in2(tt::CBIndex::c_2);  // input_c
    CircularBuffer cb_out(tt::CBIndex::c_3);

    // LLK broadcast destination CBs (dedicated per-input):
    CircularBuffer cb_llk_a(tt::CBIndex::c_4);  // broadcasted A (if used)
    CircularBuffer cb_llk_b(tt::CBIndex::c_5);  // broadcasted B (if used)
    CircularBuffer cb_llk_c(tt::CBIndex::c_6);  // broadcasted C (if used)

// Effective CBs chosen at compile-time depending on broadcast flags
#if BCAST_A
    CircularBuffer& cb_eff_a = cb_llk_a;
#else
    CircularBuffer& cb_eff_a = cb_in0;
#endif
#if BCAST_B
    CircularBuffer& cb_eff_b = cb_llk_b;
#else
    CircularBuffer& cb_eff_b = cb_in1;
#endif
#if BCAST_C
    CircularBuffer& cb_eff_c = cb_llk_c;
#else
    CircularBuffer& cb_eff_c = cb_in2;
#endif

    // Initialize binary unit for B*C path; output packer initialized with cb_out
    binary_op_init_common(cb_eff_b.get_cb_id(), cb_eff_c.get_cb_id(), cb_out.get_cb_id());

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
// 1) Prepare B and C (broadcast if required), then compute mul(B, C) -> DST[0]
// Perform LLK row broadcast for B if requested
#if BCAST_B
        cb_in1.wait_front(num_tiles_per_cycle);
        cb_llk_b.reserve_back(num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in1.get_cb_id(), cb_llk_b.get_cb_id());
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in1.get_cb_id(), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_b.get_cb_id());
        cb_llk_b.push_back(num_tiles_per_cycle);
        tile_regs_release();
        cb_in1.pop_front(num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for C if requested
#if BCAST_C
        cb_in2.wait_front(num_tiles_per_cycle);
        cb_llk_c.reserve_back(num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in2.get_cb_id(), cb_llk_c.get_cb_id());
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in2.get_cb_id(), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_c.get_cb_id());
        cb_llk_c.push_back(num_tiles_per_cycle);
        tile_regs_release();
        cb_in2.pop_front(num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for A if requested (do this before compute regs session)
#if BCAST_A
        cb_in0.wait_front(num_tiles_per_cycle);
        cb_llk_a.reserve_back(num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in0.get_cb_id(), cb_llk_a.get_cb_id());
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in0.get_cb_id(), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_a.get_cb_id());
        cb_llk_a.push_back(num_tiles_per_cycle);
        tile_regs_release();
        cb_in0.pop_front(num_tiles_per_cycle);
#endif

        // Ensure sources available
        cb_eff_a.wait_front(num_tiles_per_cycle);
        cb_eff_b.wait_front(num_tiles_per_cycle);
        cb_eff_c.wait_front(num_tiles_per_cycle);

        tile_regs_acquire();

        copy_tile_init(cb_eff_a.get_cb_id());
        copy_tile(cb_eff_a.get_cb_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_tile_init(cb_eff_b.get_cb_id());
        copy_tile(cb_eff_b.get_cb_id(), 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_tile_init(cb_eff_c.get_cb_id());
        copy_tile(cb_eff_c.get_cb_id(), 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Reserve output buffer only when ready to write
        cb_out.reserve_back(num_tiles_per_cycle);

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out.get_cb_id());

        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);
        cb_eff_a.pop_front(num_tiles_per_cycle);
        cb_eff_b.pop_front(num_tiles_per_cycle);
        cb_eff_c.pop_front(num_tiles_per_cycle);
    }
}
