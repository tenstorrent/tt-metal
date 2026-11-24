// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h. Same instruction sequence and
// access kinds as production, expressed through the HAL. The tiny-tile addrmods still come from llk_math_matmul.h as
// it is in this tree.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_matmul.h" // tile-shape-aware matmul_configure_addrmod reused for 16x32 tiny-tile support

using namespace ckernel;

namespace hal_port
{

// One ADDR_MOD slot in the units of ckernel::addr_mod_t. Bias is always zero here.
struct NoMopAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srca_cr   = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t srcb_cr   = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// The addr_mod_t set() of slot AddrModIndex: three immediate SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully
// specified.
template <std::uint8_t AddrModIndex, NoMopAddrMod M>
inline void no_mop_set_addr_mod()
{
    namespace cfg        = hal::cfg;
    constexpr cfg::Sec S = static_cast<cfg::Sec>(AddrModIndex);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, M.srca_cr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, M.srcb_cr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// The MVMUL with clear_dvalid = Release, instr_mod19 0, Dest offset 0
template <std::uint8_t AddrMode, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void no_mop_mvmul()
{
    constexpr hal::fpu::MatrixMultiply op {.address_modifier = AddrMode, .release = Release};
    static_assert(hal::fpu::get_operation<op>() == TT_OP_MVMUL(static_cast<std::uint32_t>(Release), 0, AddrMode, 0), "MVMUL encoding");
    hal::fpu::run<op>();
}

// The MVMUL releasing SrcA when reuse_a, else SrcB (production picks CLR_A or CLR_B at runtime)
template <std::uint8_t AddrMode>
inline __attribute__((always_inline)) void no_mop_mvmul_release(const bool reuse_a)
{
    if (reuse_a)
    {
        no_mop_mvmul<AddrMode, hal::fpu::SourceRelease::SrcA>();
    }
    else
    {
        no_mop_mvmul<AddrMode, hal::fpu::SourceRelease::SrcB>();
    }
}

// SETRWC(CLR_A or CLR_B, 0, 0, 0, 0, SET_ABD_F): reset SrcA/SrcB/Dest and the fidelity phase, releasing one source.
template <hal::counters::rwc::Src Flip>
inline __attribute__((always_inline)) void no_mop_reset_abd_f_release()
{
    namespace rwc = hal::counters::rwc;
    constexpr rwc::MathCounter c {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = Flip, .clear_fidelity = true};
    static_assert(c.operation() == TT_OP_SETRWC(Flip == rwc::Src::A ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F), "SETRWC encoding");
    rwc::set<c>();
}

} // namespace hal_port

// Helper functions for math fidelity
constexpr int get_math_num_fidelity_phases(const MathFidelity math_fidelity)
{
    // LoFi = 0 has 0 fidelity phases
    // HiFi2 = 1 has 1 phase, HiFi3 = 2 has 2 phases, HiFi4 = 3 has 3 phases
    return ckernel::to_underlying(math_fidelity);
}

inline void matmul_validate_no_mop_contract(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    LLK_ASSERT(
        in0_tile_r_dim == TILE_R_DIM && in0_tile_c_dim == TILE_C_DIM && in1_tile_r_dim == TILE_R_DIM && in1_tile_c_dim == TILE_C_DIM && !partial_face,
        "Blackhole custom no-mop matmul currently supports only full 32x32 tiles with partial_face disabled");
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_addrmod_no_mop(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    static_assert(THROTTLE_LEVEL >= 0 && THROTTLE_LEVEL <= 5, "THROTTLE_LEVEL must be in range [0, 5]");

    // For 16x32 tiny tiles (or any non-full-tile geometry) delegate to the tile-shape-aware regular
    // matmul address-modifier programming. That path programs the SrcB cr-reset / reduced increments
    // so the fixed no-mop replay stays confined to the tile's actual face-rows (mirrors the WH no-mop
    // path, which likewise delegates to matmul_configure_addrmod). The hardcoded full-32x32 addrmods
    // below are retained unchanged for the common full-tile case.
    const bool is_full_tile =
        (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM) && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == TILE_C_DIM) && !partial_face;
    if (!is_full_tile)
    {
        matmul_configure_addrmod<math_fidelity, THROTTLE_LEVEL, false /*ROW_MOP*/>(
            transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
        return;
    }

    constexpr bool high_fidelity     = math_fidelity != MathFidelity::LoFi;
    constexpr int fidelity_increment = high_fidelity ? 1 : 0;

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    hal_port::no_mop_set_addr_mod<ADDR_MOD_0, hal_port::NoMopAddrMod {.srcb_incr = 8, .dest_incr = 8}>();

    // reset all, increment fidelity if we have more fidelity phases
    hal_port::no_mop_set_addr_mod<
        ADDR_MOD_5,
        hal_port::NoMopAddrMod {.srca_clr = 1, .srca_cr = 1, .srcb_clr = 1, .srcb_cr = 1, .dest_clr = 1, .dest_cr = 1, .fid_incr = fidelity_increment}>();

    if constexpr (THROTTLE_LEVEL)
    {
        // reset all, including fidelity
        hal_port::no_mop_set_addr_mod<
            ADDR_MOD_6,
            hal_port::NoMopAddrMod {.srca_clr = 1, .srca_cr = 1, .srcb_clr = 1, .srcb_cr = 1, .dest_clr = 1, .dest_cr = 1, .fid_clr = 1}>();
    }

    if (transpose)
    {
        hal_port::no_mop_set_addr_mod<ADDR_MOD_1, hal_port::NoMopAddrMod {.srca_incr = 32, .srcb_cr = 1, .dest_incr = 8}>();
    }
    else
    {
        hal_port::no_mop_set_addr_mod<ADDR_MOD_1, hal_port::NoMopAddrMod {.srca_incr = 16, .srcb_cr = 1, .dest_incr = 8}>();
    }

    hal_port::no_mop_set_addr_mod<ADDR_MOD_2, hal_port::NoMopAddrMod {.srca_cr = 1, .srcb_incr = 32, .srcb_cr = 1, .dest_incr = 8}>();

    if (transpose)
    {
        hal_port::no_mop_set_addr_mod<ADDR_MOD_4, hal_port::NoMopAddrMod {.srca_incr = 16, .srca_cr = 1, .srcb_incr = 48, .srcb_cr = 1, .dest_cr = 1}>();
    }
    else
    {
        hal_port::no_mop_set_addr_mod<ADDR_MOD_4, hal_port::NoMopAddrMod {.srca_incr = 32, .srca_cr = 1, .srcb_incr = 48, .srcb_cr = 1, .dest_cr = 1}>();
    }
}

template <MathFidelity math_fidelity = MathFidelity::LoFi, int THROTTLE_LEVEL = 0>
inline void matmul_configure_addrmod_reinit(
    const bool transpose               = false,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // Reinit must restore the address-modifier contract used by replay. For tiny tiles this
    // re-establishes the shape-aware addrmods (see matmul_configure_addrmod_no_mop); for full tiles
    // it restores the hardcoded 32x32 addrmods. transpose affects ADDR_MOD_1/4, fidelity uses ADDR_MOD_5/6.
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
}

// Number of MVMULs (including the trailing fidelity/clear op) in the no-mop replay image for the given
// operand geometry. Mirrors the replay_buf_len formula in the regular matmul_configure_mop so the tiny
// (16x32 / 32x16) face-row-confined sequences record and replay the correct length.
inline std::uint32_t matmul_no_mop_replay_len(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face)
{
    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);
    return (is_in0_16x32 && is_in1_32x16) ? 4 : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : 16);
}

template <MathFidelity math_fidelity>
inline void matmul_configure_mop_custom(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3
    //
    // 16x32 / 32x16 tiny tiles record a shorter, face-row-confined MVMUL sequence (identical to the proven
    // regular matmul_configure_mop) so the replay never marches SrcB into non-existent faces; full 32x32
    // tiles keep the original 16-MVMUL 4-face walk. The no-mop path drives this via direct lltt::replay.

    constexpr bool high_fidelity = math_fidelity != MathFidelity::LoFi;

    const bool reuse_a = ct_dim >= rt_dim;

    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    const std::uint32_t replay_buf_len = matmul_no_mop_replay_len(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    hal::replay::record(
        hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len},
        // Lambda function to load reply buffer
        [high_fidelity, reuse_a, partial_face, is_in1_32x16, is_in0_16x32, is_in0_32x16, is_in1_16x32]
        {
            if (is_in1_32x16)
            {
                if (is_in0_16x32)
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A1 // srca=srca, srcb+=8,  dest=+8, bias=1
                }
                else
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A1 // srca=srca, srcb+=8,  dest=+8
                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // B1A1 // srca=0,    srcb+=8,  dest=16 (addr_mod_4), bias=0

                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B2A0 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B2A0 // srca+=16,  srcb+=8,  dest=16
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B3A1 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else if (is_in0_16x32 || is_in0_32x16)
            {
                if (partial_face)
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B0A0 // srca+=16,  srcb=0,   dest=+16
                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // B0A1 // srca+=16,  srcb+=16, dest=0 (addr_mod_4), bias=0
                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B1A2 // srca+=16,  srcb=0,   dest=+16
                }
                else
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B0A0 // srca+=16,  srcb=0,   dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A1 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // B0A1 // srca+=16/=0, srcb=16, dest=0/+=8 (addr_mod_4) // srca=0 dest+=8 if in0_32x16
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A2 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B1A2 // srca+=16,  srcb=16,  dest+=8/24 // dest+=24 if transposed
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A3 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else
            {
                hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A0 // srca=srca, srcb+=8,  dest+=8
                hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B2A0 // srca=srca, srcb+=8,  dest+=8
                hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                if (!is_in1_16x32)
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B1A3 // srca=srca, srcb+=8,  dest+=8
                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // B1A3 // srca=32,   srcb=48,  dest+=8

                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    hal_port::no_mop_mvmul<ADDR_MOD_1>(); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                    hal_port::no_mop_mvmul<ADDR_MOD_0>(); // B3A3 // srca=srca, srcb+=8,  dest+=8
                }
            }

            if constexpr (high_fidelity)
            {
                hal_port::no_mop_mvmul<ADDR_MOD_5>(); // reset srca/srcb/dest, increment phase (addr_mod_5)
            }
            else
            {
                if (reuse_a)
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_5, hal::fpu::SourceRelease::SrcA>(); // reset + increment phase (addr_mod_5), clear src A
                }
                else
                {
                    hal_port::no_mop_mvmul<ADDR_MOD_5, hal::fpu::SourceRelease::SrcB>(); // reset + increment phase (addr_mod_5), clear src B
                }
            }
        });

    // MOP template programming removed - will use direct replay calls
}

template <int Level>
void run_throttled_sequence_no_mop();

template <>
void run_throttled_sequence_no_mop<1>()
{
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_2>();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
}

template <>
void run_throttled_sequence_no_mop<2>()
{
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal_port::no_mop_mvmul<ADDR_MOD_2>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
}

template <>
void run_throttled_sequence_no_mop<3>()
{
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_2>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
}

template <>
void run_throttled_sequence_no_mop<4>()
{
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal::nop::thread();
}

template <>
void run_throttled_sequence_no_mop<5>()
{
    hal::nop::thread();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_1>();
    hal::nop::thread();
    hal::nop::thread();
    hal_port::no_mop_mvmul<ADDR_MOD_0>();
    hal::nop::thread();
    hal::nop::thread();
}

/*
 * Programming of the MOP for the case we limit matmul compute throughput
 * Done by inserting NOP instructions between MVMUL instructions of matmul kernel
 *
 * Valid range of THROTTLE_LEVEL is {1,2,3,4,5}
 * Each value corresponds to level of throttling as:
 * Level 1: throttle to 73% of max
 * Level 2: throttle to 67% of max
 * Level 3: throttle to 50% of max
 * Level 4: throttle to 40% of max
 * Level 5: throttle to 33% of max
 */
template <int THROTTLE_LEVEL>
inline void matmul_configure_mop_throttled_no_mop()
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3

    static_assert((THROTTLE_LEVEL > 0) && (THROTTLE_LEVEL <= 5), "MM throttling only enabled for THROTTLE_LEVEL={1,2,3,4,5}");

    constexpr std::uint32_t replay_buf_len = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);

    hal::replay::record<hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len}>(
        // Lambda function to load reply buffer
        [] { run_throttled_sequence_no_mop<THROTTLE_LEVEL>(); });

    // MOP template programming removed - will use direct replay calls
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_no_mop_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    if constexpr (THROTTLE_LEVEL > 0)
    {
        matmul_configure_mop_throttled_no_mop<THROTTLE_LEVEL>();
    }
    else
    {
        matmul_configure_mop_custom<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)
}

inline void _llk_math_matmul_uninit_no_mop_()
{
    // No state to restore - all states are transient or default
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_no_mop_(
    std::uint32_t dst_index,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    const bool reuse_a                = ct_dim >= rt_dim;
    const std::uint32_t t_dim         = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim       = reuse_a ? ct_dim : rt_dim; // reuse-dim
    constexpr int num_fidelity_phases = get_math_num_fidelity_phases(math_fidelity);
    constexpr bool high_fidelity      = math_fidelity != MathFidelity::LoFi;

    // Compute replay buffer length based on tile dimensions (same logic as in matmul_configure_mop_custom).
    // For 16x32 tiny tiles this is the shorter face-row-confined length (must match what was recorded at init).
    std::uint32_t replay_buf_len;
    if constexpr (THROTTLE_LEVEL > 0)
    {
        replay_buf_len = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);
    }
    else
    {
        replay_buf_len = matmul_no_mop_replay_len(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            if constexpr (THROTTLE_LEVEL > 0)
            {
                // Throttled execution
                if constexpr (THROTTLE_LEVEL > 3)
                {
                    // THROTTLE_LEVEL 4 or 5: outer_loops = 2
                    if constexpr (high_fidelity)
                    {
                        // outer loop for fidelity phases
                        for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                        {
                            // inner loop (2 iterations for standard tiles)
                            for (std::uint32_t inner = 0; inner < 2; inner++)
                            {
                                hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                                if (inner < 1)
                                {
                                    hal_port::no_mop_mvmul<ADDR_MOD_2>(); // inner loop continuation
                                }
                                else if (phase < num_fidelity_phases - 1)
                                {
                                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // last inner, not last outer
                                }
                                else
                                {
                                    hal_port::no_mop_mvmul<ADDR_MOD_5>(); // last inner, last outer
                                }
                            }
                        }
                        // Final clear
                        if (reuse_a)
                        {
                            hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::A>();
                        }
                        else
                        {
                            hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::B>();
                        }
                    }
                    else
                    {
                        // Not high fidelity, outer_loops = 2
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                            if (inner < 1)
                            {
                                hal_port::no_mop_mvmul<ADDR_MOD_2>();
                            }
                            else
                            {
                                hal_port::no_mop_mvmul<ADDR_MOD_4>();
                            }
                        }
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                            if (inner < 1)
                            {
                                hal_port::no_mop_mvmul<ADDR_MOD_2>();
                            }
                            else
                            {
                                hal_port::no_mop_mvmul_release<ADDR_MOD_5>(reuse_a);
                            }
                        }
                    }
                }
                else
                {
                    // THROTTLE_LEVEL 1, 2, or 3
                    if constexpr (high_fidelity)
                    {
                        // outer loop is num_fidelity_phases
                        for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                        {
                            for (std::uint32_t inner = 0; inner < 2; inner++)
                            {
                                hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                                if (inner < 1)
                                {
                                    hal_port::no_mop_mvmul<ADDR_MOD_4>(); // inner loop continuation
                                }
                                else if (phase < num_fidelity_phases - 1)
                                {
                                    hal_port::no_mop_mvmul<ADDR_MOD_5>(); // last inner, not last outer
                                }
                                else
                                {
                                    hal_port::no_mop_mvmul_release<ADDR_MOD_6>(reuse_a); // last inner, last outer
                                }
                            }
                        }
                    }
                    else
                    {
                        // Not high fidelity, outer_loops = 1
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                            if (inner < 1)
                            {
                                hal_port::no_mop_mvmul<ADDR_MOD_4>();
                            }
                            else
                            {
                                hal_port::no_mop_mvmul_release<ADDR_MOD_5>(reuse_a);
                            }
                        }
                    }
                }
            }
            else
            {
                // Non-throttled execution - use replay
                if constexpr (high_fidelity)
                {
                    // Replay num_fidelity_phases times
                    for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                    {
                        hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                    }
                    // Final clear after all fidelity phases
                    if (reuse_a)
                    {
                        hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::A>();
                    }
                    else
                    {
                        hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::B>();
                    }
                }
                else
                {
                    // Just replay once
                    hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
                }
            }

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::B>();
                }
                else
                {
                    hal_port::no_mop_reset_abd_f_release<hal::counters::rwc::Src::A>();
                }
            }
        }
    }
}
