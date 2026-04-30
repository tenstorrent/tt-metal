// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: PREPROCESS pre-passes stay raw LLK. Inner copy×2 + binary
// SFPU op + post-activations block migrates to V2 helper when
// num_tiles_per_cycle == 1; falls back to raw for multi-tile DEST batching.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
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
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
struct BinarySfpuMacroOp : compute_kernel_lib::BinaryOp<
                               BinarySfpuMacroOp,
                               compute_kernel_lib::Dst::D0,
                               compute_kernel_lib::Dst::D1,
                               compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;
    ALWI static void init(){
#if HAS_ACTIVATIONS(POST)
        BINARY_SFPU_INIT
#endif
    } ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) {
        BINARY_SFPU_OP(i0, i1, out_idx);
        PROCESS_POST_ACTIVATIONS(out_idx);
    }
};
}  // namespace

template <
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle) {
    using namespace ckernel;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::CopyTilePolicy;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

#if BCAST_INPUT
    constexpr auto CB_PRE_BCAST = cb_pre_rhs;
    constexpr auto CB_POST_BCAST = cb_post_rhs;
    constexpr auto CB_PRE_OTHER = cb_pre_lhs;
    constexpr auto CB_POST_OTHER = cb_post_lhs;
#else
    constexpr auto CB_PRE_BCAST = cb_pre_lhs;
    constexpr auto CB_POST_BCAST = cb_post_lhs;
    constexpr auto CB_PRE_OTHER = cb_pre_rhs;
    constexpr auto CB_POST_OTHER = cb_post_rhs;
#endif

    PREPROCESS(BCAST_OP, CB_PRE_BCAST, CB_POST_BCAST, cb_out, num_tiles_per_cycle);
    cb_wait_front(CB_POST_BCAST, num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, CB_PRE_OTHER, CB_POST_OTHER, cb_out, num_tiles_per_cycle);
        cb_wait_front(CB_POST_OTHER, num_tiles_per_cycle);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif

        // Single-tile path (num_tiles_per_cycle == 1) uses V2 helper chain.
        eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<static_cast<uint32_t>(cb_post_lhs), Dst::D0, CopyTilePolicy::NoWaitNoPop>{},
                CopyTile<static_cast<uint32_t>(cb_post_rhs), Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
                BinarySfpuMacroOp{}));

        cb_pop_front(CB_POST_OTHER, num_tiles_per_cycle);
    }
    cb_pop_front(CB_POST_BCAST, num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "binary_sfpu chain path runs one tile per chain invocation");

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    ckernel::compute_kernel_hw_startup(cb_post_lhs, cb_out);
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out>(
            tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile<cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out>(
            remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
