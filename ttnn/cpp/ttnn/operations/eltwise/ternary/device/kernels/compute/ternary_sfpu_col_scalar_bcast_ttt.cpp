// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/dataflow/dataflow_buffer.h"

ALWI void process_tile(
    uint32_t predicate_cb_id,
    uint32_t true_cb_id,
    uint32_t false_cb_id,
    uint32_t cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    DataflowBuffer predicate_dfb(predicate_cb_id);
    DataflowBuffer true_dfb(true_cb_id);
    DataflowBuffer false_dfb(false_cb_id);
    DataflowBuffer dfb_out(cb_out_id);

    // 3-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
#if BCAST_A
    predicate_dfb.wait_front(num_tiles_per_cycle);  // predicate_dfb is broadcast
#endif
#if BCAST_B
    true_dfb.wait_front(num_tiles_per_cycle);  // true_dfb is broadcast
#endif
#if BCAST_C
    false_dfb.wait_front(num_tiles_per_cycle);  // false_dfb is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        predicate_dfb.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        true_dfb.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        false_dfb.wait_front(num_tiles_per_cycle);
#endif

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy all 3 inputs to destination registers
        copy_tile_init(predicate_dfb.get_id());
        copy_tile(predicate_dfb.get_id(), 0, 0);  // predicate to reg 0, 3, 6, ...

        copy_tile_init(true_dfb.get_id());
        copy_tile(true_dfb.get_id(), 0, 1);  // true to reg 1, 4, 7, ...

        copy_tile_init(false_dfb.get_id());
        copy_tile(false_dfb.get_id(), 0, 2);  // false to reg 2, 5, 8, ...

        // Perform the ternary operation
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();

        tile_regs_wait();

        pack_tile(0, dfb_out.get_id());  // result is stored in predicate register
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        predicate_dfb.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        true_dfb.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        false_dfb.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    predicate_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B
    true_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C
    false_dfb.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb_id = tt::CBIndex::c_0;
    constexpr auto true_cb_id = tt::CBIndex::c_1;
    constexpr auto false_cb_id = tt::CBIndex::c_2;
    constexpr auto cb_out_id = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb_id, cb_out_id);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(predicate_cb_id, true_cb_id, false_cb_id, cb_out_id, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            predicate_cb_id, true_cb_id, false_cb_id, cb_out_id, remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
