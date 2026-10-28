// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice compute — RAW-LLK datapath bench (perf experiment, not the op).
//
// Same contract as the op's compute kernel: CT [cb_in, cb_out, block_tiles], RT [core_tiles, min_bits,
// max_bits], FullBlock CB sync (nominal block_tiles wait/pop/reserve/push; math/pack touch only `valid`).
//
// RAW-LLK JUSTIFICATION (helper bypassed: compute_kernel_lib::eltwise_chain — CopyTile/Clamp/Square/PackTile):
//   The chain issues one SFPU bracket per tile per op (_llk_math_eltwise_sfpu_start_: SETC16 dest offset +
//   STALLWAIT SFPU<-MATH; 4x per-face SETRWC; done) and re-inits Clamp+Square per block because the two SFPU
//   types are not "uniform" (chain_hoist_sfpu = false). This kernel exploits that DEST tiles of one half-sync
//   block are contiguous (tile k = base + 64k rows), so ONE bracket can walk 4*valid faces, and that clamp
//   and square share no SFPU state except vConstIntPrgm0 (LREG12: clamp's bound, square's RNE lsb mask),
//   so square's init can run once and only Prgm0 is restored. No helper expresses a multi-tile SFPU walk
//   or a two-op single-bracket walk.
//
// Defines (all compile-time):
//   CDO_COPY_MODE  0 = copy_tile per tile (helper-equivalent), 1 = copy_block, 2 = no copy (ABLATION ONLY: wrong out)
//   CDO_SFPU_MODE  0 = clamp_tile x valid then square_tile x valid, per-block inits (helper-equivalent)
//              1 = one bracket per op per block (clamp walk over 4*valid faces, then square walk), square init once
//              2 = one bracket per block, per face: clamp, Prgm0=1, square  (single DEST walk, two bodies)
//              3 = no SFPU (ABLATION ONLY)
//              4 = like 1, but clamp walk with bounds hoisted into LREG12/13 once per block (one LOAD/STORE per row)
//   CDO_SWAP_NOP  (SFPU_MODE 4) insert SFPNOP after each SFPSWAP (hazard probe)
//   CDO_DO_CLAMP / CDO_DO_SQUARE  (CDO_SFPU_MODE 0 only) per-op ablation toggles (ABLATION ONLY when 0)
//   CDO_PACK_MODE  0 = pack_tile per tile, 1 = pack_block
//   CDO_SFPU_INIT_PER_BLOCK (CDO_SFPU_MODE 0) 1 = re-init clamp/square every block (helper), 0 = only Prgm0 restore

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef CDO_COPY_MODE
#define CDO_COPY_MODE 0
#endif
#ifndef CDO_SFPU_MODE
#define CDO_SFPU_MODE 0
#endif
#ifndef CDO_DO_CLAMP
#define CDO_DO_CLAMP 1
#endif
#ifndef CDO_DO_SQUARE
#define CDO_DO_SQUARE 1
#endif
#ifndef CDO_PACK_MODE
#define CDO_PACK_MODE 0
#endif
#ifndef CDO_SWAP_NOP
#define CDO_SWAP_NOP 0
#endif
#ifndef CDO_SFPU_INIT_PER_BLOCK
#define CDO_SFPU_INIT_PER_BLOCK 1
#endif

#ifdef TRISC_MATH
namespace {
// One SFPU bracket over `nfaces` contiguous DEST faces starting at tile 0 of the current half.
inline void clamp_walk(uint32_t nfaces, uint32_t lo, uint32_t hi) {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < nfaces; ++f) {
        ckernel::sfpu::calculate_clamp<APPROX, 8>(lo, hi);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
inline void square_walk(uint32_t nfaces) {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < nfaces; ++f) {
        ckernel::sfpu::calculate_square<APPROX, false, 8>();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
inline void clamp_square_walk(uint32_t nfaces, uint32_t lo, uint32_t hi) {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < nfaces; ++f) {
        ckernel::sfpu::calculate_clamp<APPROX, 8>(lo, hi);
        // calculate_clamp walked the D counter through the face; SETRWC(CR_D, +0) rewinds it to the
        // face start (the carriage-return register the per-face increment is relative to).
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_D);
        sfpi::vConstIntPrgm0 = 1;  // clamp clobbered LREG12; square's RNE needs lsb mask 1
        ckernel::sfpu::calculate_square<APPROX, false, 8>();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
// Clamp walk with the bound registers loaded ONCE per block instead of twice per SFPU row:
// lo -> vConstIntPrgm0 (LREG12), hi -> vConstIntPrgm1 (LREG13); per row LOAD, SWAP(max lo), SWAP(min hi),
// STORE. Same SFPSWAP min/max ops as calculate_clamp (exact, so bit-identical); the library body
// re-issues `vConstIntPrgm0 = bound` (SFPLOADI x2 + SFPCONFIG) and a LOAD/STORE pair per bound per row.
// Restores square's Prgm0/Prgm1 afterwards.
inline void clamp_walk_hoisted(uint32_t nfaces, uint32_t lo, uint32_t hi) {
    sfpi::vConstIntPrgm0 = lo;
    sfpi::vConstIntPrgm1 = hi;
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < nfaces; ++f) {
#pragma GCC unroll 8
        for (int d = 0; d < 8; ++d) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, 9);  // VD = max(x, lo)
#if CDO_SWAP_NOP
            TTI_SFPNOP;
#endif
            TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // VD = min(., hi)
#if CDO_SWAP_NOP
            TTI_SFPNOP;
#endif
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
}
}  // namespace
#endif

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);
    const uint32_t min_bits = get_arg_val<uint32_t>(1);
    const uint32_t max_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
#if CDO_SFPU_MODE == 1 || CDO_SFPU_MODE == 2 || CDO_SFPU_MODE == 4 || (CDO_SFPU_MODE == 0 && !CDO_SFPU_INIT_PER_BLOCK)
    clamp_tile_init();
    square_tile_init();  // last: owns ADDR_MOD_6 (+4) and Prgm0..2; clamp only touches Prgm0
#endif

    MaybeDeviceZoneScope("compute_chain");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t remaining = core_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;

        cb_wait_front(cb_in, block_tiles);
        tile_regs_acquire();

#if CDO_COPY_MODE == 0
        for (uint32_t j = 0; j < valid; ++j) {
            copy_tile(cb_in, j, j);
        }
#elif CDO_COPY_MODE == 1
        copy_block(cb_in, 0, 0, valid);
#endif

#if CDO_SFPU_MODE == 0
#if CDO_SFPU_INIT_PER_BLOCK
        clamp_tile_init();
#endif
#if CDO_DO_CLAMP
        for (uint32_t j = 0; j < valid; ++j) {
            clamp_tile(j, min_bits, max_bits);
        }
#endif
#if CDO_SFPU_INIT_PER_BLOCK
        square_tile_init();
#else
        MATH((sfpi::vConstIntPrgm0 = 1));
#endif
#if CDO_DO_SQUARE
        for (uint32_t j = 0; j < valid; ++j) {
            square_tile(j);
        }
#endif
#elif CDO_SFPU_MODE == 1
        MATH((clamp_walk(4 * valid, min_bits, max_bits)));
        MATH((sfpi::vConstIntPrgm0 = 1));
        MATH((square_walk(4 * valid)));
#elif CDO_SFPU_MODE == 2
        MATH((clamp_square_walk(4 * valid, min_bits, max_bits)));
#elif CDO_SFPU_MODE == 4
        MATH((clamp_walk_hoisted(4 * valid, min_bits, max_bits)));
#if CDO_DO_SQUARE
        MATH((square_walk(4 * valid)));
#endif
#endif

        tile_regs_commit();
        cb_pop_front(cb_in, block_tiles);

        cb_reserve_back(cb_out, block_tiles);
        tile_regs_wait();
#if CDO_PACK_MODE == 0
        for (uint32_t j = 0; j < valid; ++j) {
            pack_tile(j, cb_out);
        }
#else
        pack_block(0, cb_out, valid);
#endif
        tile_regs_release();
        cb_push_back(cb_out, block_tiles);
    }
}
