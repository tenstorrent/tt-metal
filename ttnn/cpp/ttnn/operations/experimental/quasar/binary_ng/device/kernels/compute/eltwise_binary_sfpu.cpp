// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"

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
#include "api/compute/atan2.h"
#include "api/compute/binary_comp.h"
#include "api/compute/isclose.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs_id,
    tt::CBIndex cb_post_lhs_id,
    tt::CBIndex cb_pre_rhs_id,
    tt::CBIndex cb_post_rhs_id,
    tt::CBIndex cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle ISCLOSE_RT_ARG_PARAMS) {
    using namespace ckernel;

    DataflowBuffer cb_post_lhs(cb_post_lhs_id);
    DataflowBuffer cb_post_rhs(cb_post_rhs_id);
    DataflowBuffer cb_out(cb_out_id);

#if BCAST_INPUT
#define CB_PRE_BCAST cb_pre_rhs_id
#define CB_PRE_OTHER cb_pre_lhs_id
    DataflowBuffer& cb_post_bcast = cb_post_rhs;
    DataflowBuffer& cb_post_other = cb_post_lhs;
#else
#define CB_PRE_BCAST cb_pre_lhs_id
#define CB_PRE_OTHER cb_pre_rhs_id
    DataflowBuffer& cb_post_bcast = cb_post_lhs;
    DataflowBuffer& cb_post_other = cb_post_rhs;
#endif

    PREPROCESS(BCAST_OP, DataflowBuffer(CB_PRE_BCAST), cb_post_bcast, cb_out, num_tiles_per_cycle);
    cb_post_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, DataflowBuffer(CB_PRE_OTHER), cb_post_other, cb_out, num_tiles_per_cycle);
        cb_post_other.wait_front(num_tiles_per_cycle);

        cb_out.reserve_back(num_tiles_per_cycle);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs.get_id(), cb_post_lhs.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs.get_id(), i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs.get_id(), cb_post_rhs.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs.get_id(), i, i * 2 + 1);

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
#if ISCLOSE_OP
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2, rtol_bits, atol_bits);
#else
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
#endif
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out.get_id());
        }
        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);
        cb_post_other.pop_front(num_tiles_per_cycle);
    }
    cb_post_bcast.pop_front(num_tiles_per_cycle);
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

    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;
    constexpr auto cb_out_id = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs_id = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id;
    constexpr auto cb_post_rhs_id = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id;

    unary_op_init_common(cb_post_lhs_id, cb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_pre_lhs_id,
            cb_post_lhs_id,
            cb_pre_rhs_id,
            cb_post_rhs_id,
            cb_out_id,
            tile_freq,
            tile_start,
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_pre_lhs_id,
            cb_post_lhs_id,
            cb_pre_rhs_id,
            cb_post_rhs_id,
            cb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle ISCLOSE_RT_ARG_FWD);
    }
}
