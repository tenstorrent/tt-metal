// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"

#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_TILE)
// The chain forms the program factory emits into SFPU_OP_CHAIN_0_TILE: a later tile's init re-programs only the op's
// state another op wrote, square stores through ADDR_MOD_4 (free here) and keeps its rounding constants out of Prgm0-2
// when another op writes them, the reciprocal leaves Prgm0.
namespace ckernel {
#ifdef SFPU_OP_EXP_INCLUDE
template <bool approx, bool constants, bool upper_macros>
ALWI void exp_tile_chain_reinit() {
    MATH((sfpu::exp_init<approx, 0x3F800000, true, DST_ACCUM_MODE, false, constants, upper_macros>()));
}
#endif
#ifdef SFPU_OP_RECIP_INCLUDE
ALWI void recip_tile_chain_init() {
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (APPROX, DST_ACCUM_MODE, true, false)));
}
template <bool own_state>
ALWI void recip_tile_chain_reinit() {
    MATH((sfpu::recip_init<APPROX, DST_ACCUM_MODE, false, false, own_state>()));
}
ALWI void recip_tile_chain_rerecord() { MATH((sfpu::_record_reciprocal_fast_24b_5c_())); }
#endif
#ifdef SFPU_OP_RSQRT_INCLUDE
ALWI void rsqrt_tile_chain_reinit() { MATH((sfpu::rsqrt_init<APPROX>())); }
#endif
#ifdef SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
template <bool prgm_rounding>
ALWI void square_tile_chain_init() {
    MATH(llk_math_sfpu_init_once());
    MATH((sfpu::_square_init_<ADDR_MOD_4, prgm_rounding>()));
}
template <bool prgm_rounding>
ALWI void square_tile_chain(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_square,
        (APPROX, DST_ACCUM_MODE, 32, ADDR_MOD_4, prgm_rounding),
        idst,
        VectorMode::None));
}
#endif
}  // namespace ckernel
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    DataflowBuffer dfb_in(cb_input);
    DataflowBuffer dfb_out(cb_output);

    compute_kernel_hw_startup(cb_input, cb_output);
    copy_init(cb_input);
#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_BLOCK)
    // Op-major blocks: every op runs over the block's tiles before the next op, so an init or re-program the chain
    // repeats is paid once per block.
#define SFPU_OP_CHAIN_FIRST_TILE_ONLY(init) \
    if (i == 0) {                           \
        init                                \
    }
#define SFPU_OP_CHAIN_FIRST_OR_LATER_TILE(first, later) \
    if (i == 0) {                                       \
        first                                           \
    } else {                                            \
        later                                           \
    }
#define SFPU_OP_CHAIN_FOR_EACH_TILE(func) \
    for (uint32_t t = 0; t < n; ++t) {    \
        func                              \
    }
    for (uint32_t i = 0; i < num_tiles; i += SFPU_OP_CHAIN_0_BLOCK) {
        const uint32_t n = num_tiles - i < SFPU_OP_CHAIN_0_BLOCK ? num_tiles - i : SFPU_OP_CHAIN_0_BLOCK;
        tile_regs_acquire();

        dfb_in.wait_front(n);
        dfb_out.reserve_back(n);

        for (uint32_t t = 0; t < n; ++t) {
            copy_tile(cb_input, t, t);
        }
        SFPU_OP_CHAIN_0_BLOCK_OPS

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t t = 0; t < n; ++t) {
            pack_tile(t, cb_output);
        }

        dfb_in.pop_front(n);
        dfb_out.push_back(n);

        tile_regs_release();
    }
#else
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        copy_tile(cb_input, 0, 0);

#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_FUNC_0) && !defined(SFPU_OP_CHAIN_0_FUNC_1)
        // A single op's init state outlives its calls, so only the first tile runs the init.
        if (i == 0) {
            SFPU_OP_CHAIN_0_INIT_0
        }
        SFPU_OP_CHAIN_0_FUNC_0
#elif defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_TILE)
        // The host wraps the inits whose state nothing after them writes, and the inits a later tile repeats.
#define SFPU_OP_CHAIN_FIRST_TILE_ONLY(init) \
    if (i == 0) {                           \
        init                                \
    }
#define SFPU_OP_CHAIN_FIRST_OR_LATER_TILE(first, later) \
    if (i == 0) {                                       \
        first                                           \
    } else {                                            \
        later                                           \
    }
        SFPU_OP_CHAIN_0_TILE
#else
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);

        dfb_in.pop_front(1);
        dfb_out.push_back(1);

        tile_regs_release();
    }
#endif
}
