// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/nextafter.h"
#include "api/compute/logaddexp.h"
#include "api/compute/logaddexp2.h"
#include "api/compute/binary_comp.h"
#include "api/compute/isclose.h"
#include "api/compute/situ_glu.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle ISCLOSE_RT_ARG_PARAMS) {
    using namespace ckernel;
    CircularBuffer exp_cb_out(cb_out);

    // The row-broadcast operand is expanded by unary_bcast straight into its DEST slot: packing the
    // expanded tile to L1 and copying it back would round NaN to inf and -0.0 to +0.0 in a 16-bit DEST.
    // The host routes a row-broadcast operand that carries a pre-activation to the reader-broadcast kernels.
#if BCAST_INPUT  // ROW_A_COL_B
#if HAS_ACTIVATIONS(LHS)
#error "LLK row broadcast of LHS does not support LHS activations"
#endif
#define CB_PRE_BCAST cb_pre_rhs
#define CB_POST_BCAST cb_post_rhs
    constexpr auto cb_row = tt::CBIndex::c_0;
    constexpr uint32_t row_dst_offset = 0;
#else  // ROW_B_COL_A
#if HAS_ACTIVATIONS(RHS)
#error "LLK row broadcast of RHS does not support RHS activations"
#endif
#define CB_PRE_BCAST cb_pre_lhs
#define CB_POST_BCAST cb_post_lhs
    constexpr auto cb_row = tt::CBIndex::c_1;
    constexpr uint32_t row_dst_offset = 1;
#endif
    constexpr uint32_t col_dst_offset = 1 - row_dst_offset;

    CircularBuffer exp_cb_row(cb_row);
    CircularBuffer exp_cb_post_bcast(CB_POST_BCAST);

    PREPROCESS(BCAST_OP, CircularBuffer(CB_PRE_BCAST), exp_cb_post_bcast, exp_cb_out, num_tiles_per_cycle);
    exp_cb_post_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        exp_cb_row.wait_front(num_tiles_per_cycle);
        exp_cb_out.reserve_back(num_tiles_per_cycle);
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        reconfig_data_format(cb_row, cb_row);
        unary_bcast_init<BroadcastType::ROW>(cb_row);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            unary_bcast<BroadcastType::ROW>(cb_row, i, i * 2 + row_dst_offset);
        }
        reconfig_data_format_srca(cb_row, CB_POST_BCAST);
        copy_init(CB_POST_BCAST);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(CB_POST_BCAST, i, i * 2 + col_dst_offset);

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
#ifdef ISCLOSE_OP
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2, rtol_bits, atol_bits);
#else
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
#endif
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        tile_regs_release();
        exp_cb_out.push_back(num_tiles_per_cycle);
        exp_cb_row.pop_front(num_tiles_per_cycle);
    }
    exp_cb_post_bcast.pop_front(num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg_val<uint32_t>(ISCLOSE_RTOL_RT_ARG_IDX);
    const uint32_t atol_bits = get_arg_val<uint32_t>(ISCLOSE_ATOL_RT_ARG_IDX);
#endif

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1;

    // One-time hardware startup: must be the first Compute API call and run exactly once.
    compute_kernel_hw_startup(cb_post_lhs, cb_out);

#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            tile_freq,
            tile_start,
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
    }
}
