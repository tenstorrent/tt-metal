// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_copy_init.hpp"

// Operands in in_a's formats share its copy init, which leaves the SFPU state the op init set, so the inits run once.
#ifndef ARCH_QUASAR
constexpr bool init_once =
    !DST_ACCUM_MODE && copy_init_shared_with_c0<dfb::in_b>() && copy_init_shared_with_c0<dfb::in_c>();
#else
constexpr bool init_once = false;
#endif

ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle, uint32_t scalar_arg) {
    using namespace ckernel;

    DataflowBuffer dfb_in0(dfb::in_a);  // input_a
    DataflowBuffer dfb_in1(dfb::in_b);  // input_b
    DataflowBuffer dfb_in2(dfb::in_c);  // input_c
    DataflowBuffer dfb_out(dfb::out);   // output

    // 3-tensor broadcast-aware synchronization - wait for broadcast DFBs outside loop
#if BCAST_A
    dfb_in0.wait_front(num_tiles_per_cycle);  // input_a is broadcast
#endif
#if BCAST_B
    dfb_in1.wait_front(num_tiles_per_cycle);  // input_b is broadcast
#endif
#if BCAST_C
    dfb_in2.wait_front(num_tiles_per_cycle);  // input_c is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast DFBs inside loop
#if !BCAST_A
        dfb_in0.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        dfb_in1.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        dfb_in2.wait_front(num_tiles_per_cycle);
#endif

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load all three inputs into DST registers
        copy_init_unless<init_once>(dfb::in_a);
        copy_tile(dfb::in_a, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        reconfig_data_format_srca(dfb::in_a, dfb::in_b);
        copy_init_unless<init_once>(dfb::in_b);
        copy_tile(dfb::in_b, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        reconfig_data_format_srca(dfb::in_b, dfb::in_c);
        copy_init_unless<init_once>(dfb::in_c);
        copy_tile(dfb::in_c, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);
        reconfig_data_format_srca(dfb::in_c, dfb::in_a);

        // Use direct addcmul kernel: computes input_a + scalar_arg * input_b * input_c -> DST[0]
        if constexpr (!init_once) {
            TERNARY_SFPU_OP_INIT();
        }
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Pack the result from DST[0] to output
        pack_tile(0, dfb::out);

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        // Pop non-broadcast DFBs inside loop
#if !BCAST_A
        dfb_in0.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        dfb_in1.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        dfb_in2.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast DFBs outside loop
#if BCAST_A
    dfb_in0.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B
    dfb_in1.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C
    dfb_in2.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    uint32_t scalar_arg = get_arg(args::scalar_arg);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    if (num_tiles == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::in_a, dfb::out);
    copy_init(dfb::in_a);
    if constexpr (init_once) {
        TERNARY_SFPU_OP_INIT();
    }

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }

    if (remaining_iterations > 0) {
        process_tile(remaining_iterations, tile_start, num_tiles_per_cycle, scalar_arg);
    }
}
