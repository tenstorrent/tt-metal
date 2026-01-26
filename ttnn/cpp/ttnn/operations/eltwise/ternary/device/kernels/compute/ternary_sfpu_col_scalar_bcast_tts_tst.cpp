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
    tt::CBIndex tensor_cb,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle,
    uint32_t scalar) {
    using namespace ckernel;

#if BCAST_A
    cb_wait_front(predicate_cb, num_tiles_per_cycle);  // predicate_cb is broadcast
#endif
#if BCAST_B && !BCAST_C  // TTS case: true tensor is broadcast
    cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B  // TST case: false tensor is broadcast
    cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)  // Neither or both broadcast
        cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy predicate to destination register 0
        copy_tile_init(predicate_cb);
        copy_tile(predicate_cb, 0, 0);

        // Fill scalar and copy tensor based on variant
        fill_tile_init();
        copy_tile_init(tensor_cb);

        // TTS: scalar=false (reg 2), tensor=true (reg 1)
        // TST: scalar=true (reg 1), tensor=false (reg 2)
        if constexpr (get_compile_time_arg_val(1) == 0) {  // TTS: scalar is false
            // Copy true tensor to reg 1
            copy_tile(tensor_cb, 0, 1);
            // Fill false scalar to reg 2
#ifdef FILL_WITH_VALUE_FLOAT
            const auto scalar_val = reinterpret_cast<const float*>(&scalar);
            FILL_LLK(2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(2, scalar);
#endif
        } else {  // TST: scalar is true
                  // Fill true scalar to reg 1
#ifdef FILL_WITH_VALUE_FLOAT
            const auto scalar_val = reinterpret_cast<const float*>(&scalar);
            FILL_LLK(1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(1, scalar);
#endif
            // Copy false tensor to reg 2
            copy_tile(tensor_cb, 0, 2);
        }

        // Perform the ternary operation
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();

        tile_regs_wait();

        pack_tile(0, cb_out);  // result is stored in predicate register
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)  // Neither or both broadcast
        cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B && !BCAST_C  // TTS case: true tensor is broadcast
    cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B  // TST case: false tensor is broadcast
    cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);  // 1=TST, 0=TTS

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto tensor_cb = tt::CBIndex::c_1;  // Either true (TTS) or false (TST) tensor
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(predicate_cb, tensor_cb, cb_out, tile_freq, tile_start, num_tiles_per_cycle, scalar_value);
    }

    if (remaining_iterations > 0) {
        process_tile(
            predicate_cb, tensor_cb, cb_out, remaining_iterations, tile_start, num_tiles_per_cycle, scalar_value);
    }
}
}  // namespace NAMESPACE
