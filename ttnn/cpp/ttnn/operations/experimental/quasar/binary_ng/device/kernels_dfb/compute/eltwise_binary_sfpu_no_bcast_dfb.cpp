// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) SFPU compute kernel for binary_ng's no-broadcast binary op.
//
// 1:1 mirror of the CircularBuffer kernels/compute/eltwise_binary_sfpu_no_bcast.cpp, with the
// CB->DFB swap. Handles the SFPU binary ops (int add/sub/mul, compares, bitwise/shift, div/remainder,
// quant, gcd/lcm, xlogy, atan2, isclose, max/min, …) using the same define machinery the descriptor
// factory builds: BINARY_SFPU_OP / BINARY_SFPU_INIT, HAS_ACTIVATIONS / PREPROCESS /
// PROCESS_POST_ACTIVATIONS, ISCLOSE_* runtime args. Operand ids come from DFBAccessor's
// `operator uint32_t()`. Layout-agnostic (reader/writer absorb sharded/interleaved/mixed).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
// SFPU op headers that ARE ported to Quasar (internally ARCH_QUASAR-aware). bf16 multiply/divide use
// eltwise_binary_sfpu.h; the rest cover the other Quasar-supported SFPU binary ops.
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/binary_comp.h"
// SFPU op headers NOT yet ported to Quasar: they unconditionally pull in WH/BH-only ckernel impls or
// symbols (InstrModLoadStore, DataFormat::UInt32, ckernel_sfpu_div_int32_floor.h, ...) absent from the
// Quasar ckernel tree, so they only compile off-Quasar. The float SFPU binary ops this kernel runs on
// Quasar never use them; exclude them there (mirrors the ARCH_QUASAR guard in eltwise_binary_sfpu.h).
#ifndef ARCH_QUASAR
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/isclose.h"
#endif

#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu_dfb.hpp"

FORCE_INLINE void process_sfpu_tiles(
    uint32_t n,
    uint32_t dfb_pre_lhs_id,
    uint32_t dfb_post_lhs_id,
    uint32_t dfb_pre_rhs_id,
    uint32_t dfb_post_rhs_id,
    uint32_t dfb_out_id ISCLOSE_RT_ARG_PARAMS) {
    DataflowBuffer dfb_post_lhs(dfb_post_lhs_id);
    DataflowBuffer dfb_post_rhs(dfb_post_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);

    PREPROCESS(LHS, dfb_pre_lhs_id, dfb_post_lhs_id, dfb_out_id, n);
    dfb_post_lhs.wait_front(n);

    PREPROCESS(RHS, dfb_pre_rhs_id, dfb_post_rhs_id, dfb_out_id, n);
    dfb_post_rhs.wait_front(n);

    dfb_out.reserve_back(n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif

    tile_regs_acquire();
#ifdef ARCH_QUASAR
    // Quasar's copy_tile_to_dst_init_short_with_dt is a no-op and cannot switch which operand the
    // unpacker reads, so use copy_tile_to_dst_init_short (which reprograms the unpacker descriptor)
    // to point at each operand before its copy_tile loop. matches_metal_v2_slice requires lhs and rhs
    // to share a data format, so the data-format reconfig the WH/BH _with_dt path performs is not needed.
    copy_tile_to_dst_init_short(dfb_post_lhs_id);
#else
    copy_tile_to_dst_init_short_with_dt(dfb_post_rhs_id, dfb_post_lhs_id);
#endif
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(dfb_post_lhs_id, i, i * 2);
    }
#ifdef ARCH_QUASAR
    copy_tile_to_dst_init_short(dfb_post_rhs_id);
#else
    copy_tile_to_dst_init_short_with_dt(dfb_post_lhs_id, dfb_post_rhs_id);
#endif
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(dfb_post_rhs_id, i, i * 2 + 1);
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
        pack_tile(i * 2, dfb_out_id);
    }
    tile_regs_release();

    dfb_out.push_back(n);
    dfb_post_lhs.pop_front(n);
    dfb_post_rhs.pop_front(n);
}

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg(args::rtol_bits);
    const uint32_t atol_bits = get_arg(args::atol_bits);
#endif

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);
    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    // post_lhs/post_rhs DFBs (c_3/c_4) exist only when that operand has an activation chain, so the
    // factory binds dfb::post_lhs / dfb::post_rhs only in that case. Guard the references with #if (not
    // a ?: ) so the no-activation build, where the accessors are unbound, still compiles. When absent,
    // the post id aliases the pre id (PREPROCESS is a no-op and the binary op reads pre directly).
#if HAS_ACTIVATIONS(LHS)
    constexpr uint32_t dfb_post_lhs_id = static_cast<uint32_t>(dfb::post_lhs);
#else
    constexpr uint32_t dfb_post_lhs_id = dfb_pre_lhs_id;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr uint32_t dfb_post_rhs_id = static_cast<uint32_t>(dfb::post_rhs);
#else
    constexpr uint32_t dfb_post_rhs_id = dfb_pre_rhs_id;
#endif

    unary_op_init_common(dfb_post_lhs_id, dfb_out_id);
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
            dfb_pre_lhs_id,
            dfb_post_lhs_id,
            dfb_pre_rhs_id,
            dfb_post_rhs_id,
            dfb_out_id ISCLOSE_RT_ARG_FWD);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_sfpu_tiles(
            remainder, dfb_pre_lhs_id, dfb_post_lhs_id, dfb_pre_rhs_id, dfb_post_rhs_id, dfb_out_id ISCLOSE_RT_ARG_FWD);
    }
}
