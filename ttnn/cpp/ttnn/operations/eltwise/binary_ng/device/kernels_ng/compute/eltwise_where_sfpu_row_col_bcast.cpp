// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_in0,
    tt::CBIndex cb_in1,
    tt::CBIndex cb_out,
    const uint32_t scalar_value,
    const float* scalar_val,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

#if BCAST_INPUT  // ROW_A_COL_B
                 // BCAST_INPUT == 1 : input B ( true or false tensor) is broadcasted
#define CB_BCAST cb_in1
#define CB_OTHER cb_in0
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_left = tt::CBIndex::c_5;
    auto cb_right = cb_in1;
#else  // ROW_B_COL_A
    // BCAST_INPUT == 0 : input A (condition tensor)  is broadcasted
#define CB_BCAST cb_in0
#define CB_OTHER cb_in1
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    auto cb_left = cb_in0;
    constexpr auto cb_right = tt::CBIndex::c_6;
#endif

    unary_op_init_common(cb_left, cb_out);
    BINARY_SFPU_INIT

    cb_wait_front(CB_BCAST, num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(CB_OTHER, num_tiles_per_cycle);
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(CB_OTHER, cb_llk_post);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(CB_OTHER, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();

        cb_pop_front(CB_OTHER, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_llk_post, num_tiles_per_cycle);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_left);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_left, i, i * 3);
        }
        copy_tile_to_dst_init_short(cb_right);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
// TTS: tensor is true value, goes to dst_reg 1
#if WHERE_TTS
            copy_tile(cb_right, i, i * 3 + 1);  // Copy true tensor to dst_reg 1
            fill_tile_init();
// TTS: scalar is false value, goes to dst_reg 2
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 2, scalar_value);
#endif
#endif

// TST: tensor is false value, goes to dst_reg 2
#if WHERE_TST
            copy_tile(cb_right, i, i * 3 + 2);  // Copy false tensor to dst_reg 2
            fill_tile_init();
// TST: scalar is true value, goes to dst_reg 1
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 1, scalar_value);
#endif
#endif

            BINARY_SFPU_OP(i * 3, i * 3 + 1, i * 3 + 2, i * 3);
        }

        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 3, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_llk_post, num_tiles_per_cycle);
    }
    cb_pop_front(CB_BCAST, num_tiles_per_cycle);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const float* scalar_value_ptr = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_in0, cb_in1, cb_out, scalar_value, scalar_value_ptr, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_in0,
            cb_in1,
            cb_out,
            scalar_value,
            scalar_value_ptr,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
