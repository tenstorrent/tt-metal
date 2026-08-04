// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/dataflow/dataflow_buffer.h"

ALWI void process_tile(
    uint32_t predicate_cb_id,
    uint32_t tensor_cb_id,
    uint32_t cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle,
    uint32_t scalar) {
    using namespace ckernel;

    DataflowBuffer predicate_dfb(predicate_cb_id);
    DataflowBuffer tensor_dfb(tensor_cb_id);
    DataflowBuffer dfb_out(cb_out_id);

#if BCAST_A
    predicate_dfb.wait_front(num_tiles_per_cycle);  // predicate_dfb is broadcast
#endif
#if BCAST_B && !BCAST_C  // TTS case: true tensor is broadcast
    tensor_dfb.wait_front(num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B  // TST case: false tensor is broadcast
    tensor_dfb.wait_front(num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        predicate_dfb.wait_front(num_tiles_per_cycle);
#endif
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)  // Neither or both broadcast
        tensor_dfb.wait_front(num_tiles_per_cycle);
#endif

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy predicate to destination register 0
        copy_tile_init(predicate_dfb.get_id());
        copy_tile(predicate_dfb.get_id(), 0, 0);

        // Fill scalar and copy tensor based on variant
        fill_tile_init();
        copy_tile_init(tensor_dfb.get_id());

        // TTS: scalar=false (reg 2), tensor=true (reg 1)
        // TST: scalar=true (reg 1), tensor=false (reg 2)
        if constexpr (get_compile_time_arg_val(1) == 0) {  // TTS: scalar is false
            // Copy true tensor to reg 1
            copy_tile(tensor_dfb.get_id(), 0, 1);
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
            copy_tile(tensor_dfb.get_id(), 0, 2);
        }

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
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)  // Neither or both broadcast
        tensor_dfb.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    predicate_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B && !BCAST_C  // TTS case: true tensor is broadcast
    tensor_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B  // TST case: false tensor is broadcast
    tensor_dfb.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);  // 1=TST, 0=TTS

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb_id = tt::CBIndex::c_0;
    constexpr auto tensor_cb_id = tt::CBIndex::c_1;  // Either true (TTS) or false (TST) tensor
    constexpr auto cb_out_id = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb_id, cb_out_id);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            predicate_cb_id, tensor_cb_id, cb_out_id, tile_freq, tile_start, num_tiles_per_cycle, scalar_value);
    }

    if (remaining_iterations > 0) {
        process_tile(
            predicate_cb_id,
            tensor_cb_id,
            cb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle,
            scalar_value);
    }
}
