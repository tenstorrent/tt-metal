// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex predicate_cb,
    tt::CBIndex true_cb,
    tt::CBIndex cb_out,
    uint32_t false_scalar,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // 2-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
    // (false is scalar, so no CB for it)
#if BCAST_PRED
    cb_wait_front(predicate_cb, num_tiles_per_cycle);  // predicate_cb is broadcast
#endif
#if BCAST_TRUE
    cb_wait_front(true_cb, num_tiles_per_cycle);  // true_cb is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_PRED
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_TRUE
        cb_wait_front(true_cb, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy predicate to destination register 0
        copy_tile_init(predicate_cb);
        copy_tile(predicate_cb, 0, 0);  // predicate to reg 0

        // Copy true tensor to destination register 1
        copy_tile_init(true_cb);
        copy_tile(true_cb, 0, 1);  // true to reg 1

        // Fill scalar false value to destination register 2
        fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
        const auto false_value = reinterpret_cast<const float*>(&false_scalar);
        FILL_LLK(2, *false_value);
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(2, false_scalar);
#endif

        // Perform the where operation: where(predicate, true, false)
        where_tile_init();
        WHERE_LLK(0, 1, 2, 0);

        tile_regs_commit();

        tile_regs_wait();

        pack_tile(0, cb_out);  // result is stored in register 0
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_PRED
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_TRUE
        cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_PRED
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_TRUE
    cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t false_scalar = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto true_cb = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(predicate_cb, true_cb, cb_out, false_scalar, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            predicate_cb, true_cb, cb_out, false_scalar, remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
