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
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"
#include "api/compute/isclose.h"
#include "api/compute/situ_glu.h"
#include "api/compute/atan2.h"
#include "api/compute/nextafter.h"
#include "api/compute/logaddexp.h"
#include "api/compute/logaddexp2.h"
#include "api/compute/bcast.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg_val<uint32_t>(ISCLOSE_RTOL_RT_ARG_IDX);
    const uint32_t atol_bits = get_arg_val<uint32_t>(ISCLOSE_ATOL_RT_ARG_IDX);
#endif

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_out = tt::CBIndex::c_2;
    CircularBuffer exp_cb_out(cb_out);

    // Invariant of every SFPU LLK-broadcast kernel (binary, where, ternary TTT): the broadcast operand goes from
    // L1 through unary_bcast into its DEST slot with no pack in between, so NaN and -0.0 reach the SFPU op as
    // they are; a bf16 pack from a 16-bit DEST would turn them into inf and +0.0. A pre-activation on that
    // operand would need such a pack, so the host sends those cases to the reader-broadcast kernels.
#if SRC_BCAST
#if HAS_ACTIVATIONS(LHS)
#error "LLK row broadcast of LHS does not support LHS activations"
#endif
    constexpr auto cb_bcast = tt::CBIndex::c_0;
    constexpr auto cb_other = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1;
    constexpr uint32_t bcast_dst_offset = 0;
    constexpr uint32_t other_dst_offset = 1;
#endif
#if SRC_BCAST_B
#if HAS_ACTIVATIONS(RHS)
#error "LLK row broadcast of RHS does not support RHS activations"
#endif
    constexpr auto cb_bcast = tt::CBIndex::c_1;
    constexpr auto cb_other = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    constexpr uint32_t bcast_dst_offset = 1;
    constexpr uint32_t other_dst_offset = 0;
#endif
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1;

    CircularBuffer exp_cb_bcast(cb_bcast);
    CircularBuffer exp_cb_other(cb_other);

    compute_kernel_hw_startup(cb_post_lhs, cb_out);
    copy_init(cb_post_lhs);
#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, CircularBuffer(tt::CBIndex::c_0), CircularBuffer(cb_post_lhs), exp_cb_out, num_tiles_per_cycle);
        PREPROCESS(RHS, CircularBuffer(tt::CBIndex::c_1), CircularBuffer(cb_post_rhs), exp_cb_out, num_tiles_per_cycle);
        exp_cb_bcast.wait_front(num_tiles_per_cycle);
        exp_cb_other.wait_front(num_tiles_per_cycle);

        exp_cb_out.reserve_back(num_tiles_per_cycle);
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        reconfig_data_format(cb_bcast, cb_bcast);
        unary_bcast_init<BroadcastType::ROW>(cb_bcast);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            unary_bcast<BroadcastType::ROW>(cb_bcast, i, i * 2 + bcast_dst_offset);
        }
        reconfig_data_format_srca(cb_bcast, cb_other);
        copy_init(cb_other);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_other, i, i * 2 + other_dst_offset);

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
        exp_cb_bcast.pop_front(num_tiles_per_cycle);
        exp_cb_other.pop_front(num_tiles_per_cycle);
    }
}
