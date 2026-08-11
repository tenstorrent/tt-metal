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
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
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
FORCE_INLINE void process_sfpu_tiles(
    uint32_t n,
    uint32_t cb_pre_lhs_id,
    uint32_t cb_post_lhs_id,
    uint32_t cb_pre_rhs_id,
    uint32_t cb_post_rhs_id,
    uint32_t cb_out_id ISCLOSE_RT_ARG_PARAMS) {
    DataflowBuffer cb_post_lhs(cb_post_lhs_id);
    DataflowBuffer cb_post_rhs(cb_post_rhs_id);
    DataflowBuffer cb_out(cb_out_id);

    PREPROCESS(LHS, DataflowBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
    cb_post_lhs.wait_front(n);

    PREPROCESS(RHS, DataflowBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, n);
    cb_post_rhs.wait_front(n);

    cb_out.reserve_back(n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_post_rhs.get_id(), cb_post_lhs.get_id());
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_post_lhs.get_id(), i, i * 2);
    }
    copy_tile_to_dst_init_short_with_dt(cb_post_lhs.get_id(), cb_post_rhs.get_id());
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_post_rhs.get_id(), i, i * 2 + 1);
#if HAS_ACTIVATIONS(POST)
        BINARY_SFPU_INIT;
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
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i * 2, cb_out.get_id());
    }
    tile_regs_release();

    cb_out.push_back(n);
    cb_post_lhs.pop_front(n);
    cb_post_rhs.pop_front(n);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg_val<uint32_t>(ISCLOSE_RTOL_RT_ARG_IDX);
    const uint32_t atol_bits = get_arg_val<uint32_t>(ISCLOSE_ATOL_RT_ARG_IDX);
#endif

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

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

    // Process full chunks
    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_sfpu_tiles(
            num_tiles_per_cycle,
            cb_pre_lhs_id,
            cb_post_lhs_id,
            cb_pre_rhs_id,
            cb_post_rhs_id,
            cb_out_id ISCLOSE_RT_ARG_FWD);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_sfpu_tiles(
            remainder, cb_pre_lhs_id, cb_post_lhs_id, cb_pre_rhs_id, cb_post_rhs_id, cb_out_id ISCLOSE_RT_ARG_FWD);
    }
}
