// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/addcmul.h"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_in0,
    tt::CBIndex cb_in1,
    tt::CBIndex cb_in2,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle,
    uint32_t scalar_arg) {
    using namespace ckernel;

    // 3-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
#if BCAST_A
    cb_wait_front(cb_in0, num_tiles_per_cycle);  // input_a is broadcast
#endif
#if BCAST_B
    cb_wait_front(cb_in1, num_tiles_per_cycle);  // input_b is broadcast
#endif
#if BCAST_C
    cb_wait_front(cb_in2, num_tiles_per_cycle);  // input_c is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Load all three inputs into DST registers
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        // Use direct addcmul kernel: computes input_a + scalar_arg * input_b * input_c -> DST[0]
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;  // output

    unary_op_init_common(cb_in0, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_in0, cb_in1, cb_in2, cb_out, tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_in0, cb_in1, cb_in2, cb_out, remaining_iterations, tile_start, num_tiles_per_cycle, scalar_arg);
    }
}
}  // namespace NAMESPACE
