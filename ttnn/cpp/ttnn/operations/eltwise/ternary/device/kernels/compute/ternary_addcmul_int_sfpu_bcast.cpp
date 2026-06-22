// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/add_int_sfpu.h"
#include "api/dataflow/circular_buffer.h"

ALWI void process_tile(
    tt::CBIndex cb_in0_id,
    tt::CBIndex cb_in1_id,
    tt::CBIndex cb_in2_id,
    tt::CBIndex cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle,
    uint32_t scalar_arg) {
    using namespace ckernel;

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_in2(cb_in2_id);
    CircularBuffer cb_out(cb_out_id);

    // 3-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
#if BCAST_A
    cb_in0.wait_front(num_tiles_per_cycle);  // input_a is broadcast
#endif
#if BCAST_B
    cb_in1.wait_front(num_tiles_per_cycle);  // input_b is broadcast
#endif
#if BCAST_C
    cb_in2.wait_front(num_tiles_per_cycle);  // input_c is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        cb_in0.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_in1.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_in2.wait_front(num_tiles_per_cycle);
#endif

        cb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load all three inputs into DST registers
        copy_tile_init(cb_in0.get_cb_id());
        copy_tile(cb_in0.get_cb_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_tile_init(cb_in1.get_cb_id());
        copy_tile(cb_in1.get_cb_id(), 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_tile_init(cb_in2.get_cb_id());
        copy_tile(cb_in2.get_cb_id(), 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        fill_tile_init();
        fill_tile_int<ADDCMUL_DATA_FORMAT>(3, scalar_arg);

        mul_int_tile_init<ADDCMUL_DATA_FORMAT>();
        mul_int_tile<ADDCMUL_DATA_FORMAT>(3, 1, 3);
        mul_int_tile<ADDCMUL_DATA_FORMAT>(3, 2, 2);

        add_int_tile_init();
        add_int_tile<ADDCMUL_DATA_FORMAT>(0, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out.get_cb_id());

        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        cb_in0.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_in1.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_in2.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    cb_in0.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_in1.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_in2.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0_id = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1_id = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2_id = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out_id = tt::CBIndex::c_3;  // output

    unary_op_init_common(cb_in0_id, cb_out_id);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_in0_id, cb_in1_id, cb_in2_id, cb_out_id, tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_in0_id,
            cb_in1_id,
            cb_in2_id,
            cb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle,
            scalar_arg);
    }
}
