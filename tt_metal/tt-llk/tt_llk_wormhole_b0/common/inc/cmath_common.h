// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// #include "kernel_types.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"
#include "ckernel_template.h"
#include "llk_defs.h"

#ifndef SFPU_OP_PARAM
#define SFPU_OP_PARAM 0
#endif

#ifndef FUSE_SQRT_RECIP
#define FUSE_SQRT_RECIP 0
#endif

using namespace ckernel;

namespace ckernel::math
{

constexpr std::uint32_t replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                                // first 16 for sfpu, next 16 for fpu

// ---------------------------------------------------------------------------------------------
// Src zero-substitution flag (ALU_ACC_CTRL_Zero_Flag_disabled_src).
//
// The flag is a math-ALU concern: only read by MOVA2D/MOVB2D/MOVB2A/MVMUL/ELWADD/ELWMUL (the math
// thread), never by the SFPU. Crucially, NO instruction changes it as a side effect -- it moves only
// on an explicit cfg_reg_rmw. So the math thread owns it and we simply track its real value: each op
// sets the value it needs, and an already-correct value is a no-op (skip the pipe-draining STALLWAIT +
// RMW). src_zero_flag_hw caches that physical value (0xff = unknown, only at power-on).
//
// What each op wants:
//   FP compute (matmul / eltwise-binary / reduce compute-phase) and format reconfigs -> the
//     operand-driven value (keep for the int formats that require it, flush otherwise; see
//     ckernel::requires_disabled_src_zero_flag). The cached operand formats feed this.
//   Data-movement (datacopy / copy_init / transpose_dest / reduce mov-phase) -> keep (1), so bf16 -0.0
//     (which the SFPU sign ops read back out of DEST) and 16b/32b int datums pass through faithfully.
//
// Canonical (non-experimental) LLKs only touch the flag from math-thread code, so the tracked value
// stays coherent. A raw cfg write that bypasses the setter must call _invalidate_src_zero_flag_state_().
// ---------------------------------------------------------------------------------------------
static std::uint32_t src_zero_flag_hw       = 0xff; // last value written to the flag; 0xff = unknown
static std::uint32_t src_zero_flag_srca_fmt = 0xff; // cached operand formats feeding the compute default
static std::uint32_t src_zero_flag_srcb_fmt = 0xff;

// The one writer. Out-of-line so the STALLWAIT + RMW exist in a single copy (code size — a matmul
// kernel otherwise overflows its slot).
inline __attribute__((noinline)) void _apply_src_zero_flag_(const std::uint32_t value)
{
    src_zero_flag_hw = value;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(value);
}

// Set the flag to `disable`; skip if it already holds that value. The check is inlined at every call
// site so hot loops (whose operand formats change but map to the same flag value) stay call-free; only
// a genuine value change pays the out-of-line write.
inline void _configure_src_zero_flag_(const bool disable)
{
    const std::uint32_t value = disable ? 1u : 0u;
    if (src_zero_flag_hw == value)
    {
        return;
    }
    _apply_src_zero_flag_(value);
}

// A kernel tight on program-config space (e.g. ring-joint SDPA, which reconfigs ~30x) can
// #define LLK_ZEROFLAG_OUTLINE before its includes to force this configurator out-of-line -- one copy
// called from each reconfig/init site instead of an inlined fast path at every one, trading a call for
// code size. Perf-critical kernels (groupnorm welford) leave it inlined (the default).
#ifdef LLK_ZEROFLAG_OUTLINE
#define LLK_ZEROFLAG_DEFAULT_ATTR __attribute__((noinline))
#else
#define LLK_ZEROFLAG_DEFAULT_ATTR
#endif

// FP compute / format reconfig: the flag follows the operand formats. Reads the cached SrcA/SrcB formats
// -- maintained by the reconfig sites, the only places the SrcA/SrcB format actually changes -- and applies
// the operand-driven value, skipping the pipe-draining write when the flag already holds it (the steady
// state in hot loops). Takes no format args and stores nothing, so the inlined fast path at every init site
// stays tiny AND the caches can never diverge (they are refreshed on every format change, independent of
// whether the resulting flag value changed).
// TODO(tt-metal#53652): the flag must be CLEARED for all FPU compute; once that lands this collapses to
// _configure_src_zero_flag_(false) and requires_disabled_src_zero_flag() / the format cache are dropped.
inline LLK_ZEROFLAG_DEFAULT_ATTR void _configure_default_zero_flag_state_()
{
    const std::uint32_t value = requires_disabled_src_zero_flag(src_zero_flag_srca_fmt, src_zero_flag_srcb_fmt) ? 1u : 0u;
    if (src_zero_flag_hw == value)
    {
        return;
    }
    _apply_src_zero_flag_(value);
}

// Data-movement ops keep the flag set so values pass through faithfully (bf16 -0.0, 16b/32b ints).
inline void _configure_preserve_zero_flag_state_()
{
    _configure_src_zero_flag_(true);
}

// Datacopy zero-flag, chosen by the source (SrcA) format. Default is preserve (keep) so bf16 -0.0 and
// 16-bit integer datums survive the move. Exception: fp8 (e4m3 / e5m2) sources widen into a SrcA datum
// whose zero carries a nonzero high residual, so preserve would read it back as ~2^-15; those must be
// flushed (zero-substituted) to produce a clean 0. Src format carries extra high bits, so mask to 0x1F.
inline void _configure_copy_zero_flag_state_(const std::uint32_t src_dst_format)
{
    const std::uint32_t fmt = src_dst_format & 0x1F;
    // Wormhole fp8 is Lf8 (e5m2) only; Fp8_e4m3 is a Blackhole-only DataFormat.
    const bool flush_fp8 = (fmt == static_cast<std::uint32_t>(DataFormat::Lf8));
    _configure_src_zero_flag_(!flush_fp8);
}

// After a raw cfg write that bypassed the setter, mark the tracked value unknown so the next
// _configure_ re-applies from a known baseline.
inline void _invalidate_src_zero_flag_state_()
{
    src_zero_flag_hw = 0xff;
}

inline void reset_counters(const std::uint32_t setrwc)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
}

inline void incr_counters(const std::uint32_t incr_a, const std::uint32_t incr_b, const std::uint32_t incr_d, const std::uint32_t incr_cr)
{
    TT_INCRWC(incr_cr, incr_d, incr_b, incr_a);
}

// MOVD2A/MOVD2B write SrcA/SrcB from Dest, so they fall outside the Src auto-wait, which covers
// only instructions that read Src. Gate the row moves on the target bank's DVALID, and drain
// in-flight math so the Dest values those moves read back have settled.
inline void srca_bank_wait()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
}

inline void srcb_bank_wait()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
}

inline void move_d2a_fixed_face(const std::uint8_t addrmod)
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD); // MOVD2A for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, addrmod, p_movd2a::MOV_4_ROWS, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, addrmod, p_movd2a::MOV_4_ROWS, 4);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, addrmod, p_movd2a::MOV_4_ROWS, 8);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, addrmod, p_movd2a::MOV_4_ROWS, 12);
}

inline void move_d2b_fixed_face(const std::uint8_t addrmod)
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD); // MOVD2B for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, addrmod, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, addrmod, p_movd2b::MOV_4_ROWS, 4);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, addrmod, p_movd2b::MOV_4_ROWS, 8);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, addrmod, p_movd2b::MOV_4_ROWS, 12);
}

inline void move_d2a_row_broadcast_fixed_face(const std::uint8_t addrmod)
{
    // MOVD2A does not auto-wait for SrcA[MatrixUnit.SrcABank].AllowedClient == MatrixUnit, so gate on SRCA_VLD
    // before the row moves (mirrors move_d2b_fixed_face's SRCB_VLD wait). See llk_math_transpose_dest.h. tt-llk#1664.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
    // // Seems to make things 200 clocks slower. Really shouldn't though.
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 1, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 2, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 3, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 5, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 6, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 7, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 9, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 10, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 11, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 13, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 14, addrmod, p_movd2a::MOV_1_ROW, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 15, addrmod, p_movd2a::MOV_1_ROW, 0);
}

inline void move_a2d_fixed_face(const std::uint8_t addrmod)
{
    TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, addrmod, p_mova2d::MOV_8_ROWS, 0);
    TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, addrmod, p_mova2d::MOV_8_ROWS, 0);
}

template <std::uint32_t SrcReg>
inline void wait_bank_valid()
{
    if constexpr (SrcReg == Srcs::SrcA)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
    }
}

template <std::uint32_t SrcReg>
inline void clear_bank_valid()
{
    if constexpr (SrcReg == Srcs::SrcA)
    {
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_A);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_B);
    }
}

inline void wait_math_semaphores()
{
    // wait while math semaphore is on max, no room to write math results
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX);
}

inline void set_math_semaphores()
{
    // Tell packer that it has something to pack
    t6_semaphore_post<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
}

inline void math_unpack_to_dest_math_ready()
{
    t6_semaphore_wait_on_max<p_stall::STALL_SYNC>(semaphore::MATH_DONE);
    t6_semaphore_post<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::MATH_DONE);
    while (semaphore_read(semaphore::MATH_DONE) == 0)
    {
    }
    semaphore_get(semaphore::MATH_DONE);
}

inline void math_unpack_to_dest_tile_ready()
{
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::UNPACK_TO_DEST);
    t6_semaphore_get<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::UNPACK_TO_DEST);
}

template <DstTileShape tile_shape, UnpackDestination unpack_destination>
inline void set_dst_write_addr(std::uint32_t tile_index)
{
    static_assert(
        tile_shape == DstTileShape::Tile32x32 || tile_shape == DstTileShape::Tile32x16 || tile_shape == DstTileShape::Tile16x16, "Invalid tile shape");
    static_assert(DstTileShape::Tile32x32 == 0, "DstTileShape::Tile32x32 must equal 0");
    static_assert(DstTileShape::Tile32x16 == 1, "DstTileShape::Tile32x16 must equal 1");
    static_assert(DstTileShape::Tile16x16 == 2, "DstTileShape::Tile16x16 must equal 2");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x32] == 6, "DstTileSizeLog2[Tile32x32] must equal 6");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x16] == 5, "DstTileSizeLog2[Tile32x16] must equal 5");
    static_assert(DstTileSizeLog2[DstTileShape::Tile16x16] == 4, "DstTileSizeLog2[Tile16x16] must equal 4");

    std::uint32_t dst_index = tile_index << DstTileSizeLog2[tile_shape];
    dst_index               = dst_index + get_dest_buffer_base();
    if constexpr (unpack_destination == UnpackDestination::DestReg)
    {
        mailbox_write(ThreadId::UnpackThreadId, dst_index); // Send to unpacker
    }
    else
    {
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
    }
}

// Programming a dst write addr offset that gets added to base
//
inline void clear_dst_reg_addr()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

inline void set_addr_mod_base()
{
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1); // set addr mod base (use addr mods 4..7)
}

inline void clear_addr_mod_base()
{
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0); // clear addr mod base (use addr mods 0..3)
}

template <std::uint32_t num_rows = 8>
inline void inc_dst_addr()
{
    static_assert(num_rows <= 15, "num_rows must be <= 15");
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, num_rows, 0, 0, p_setrwc::SET_D);
}

inline void math_dest_wait()
{
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX);
}

inline void dest_section_flip()
{
    update_dest_offset_id();
    std::uint32_t base_addr = get_dest_buffer_base();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}

template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    }
    else
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, DEST_REGISTER_HALF_SIZE);
    }
}

inline constexpr bool is_high_fidelity(const MathFidelity math_fidelity_desc)
{
    return math_fidelity_desc != MathFidelity::LoFi;
}

inline constexpr bool is_32bit_input(const std::uint32_t src_format, const std::uint32_t dst_format)
{
    const std::uint32_t input_df  = masked_data_format(src_format);
    const std::uint32_t output_df = masked_data_format(dst_format);

    return ((input_df == to_underlying(DataFormat::Int32)) || (input_df == to_underlying(DataFormat::Float32))) &&
           ((output_df == to_underlying(DataFormat::Int32)) || (output_df == to_underlying(DataFormat::Float32)));
}

inline constexpr int get_math_num_fidelity_phases(const int math_fidelity_desc)
{
    return (math_fidelity_desc & 0x7);
}

inline constexpr int get_math_fidelity_increment(const int math_fidelity_desc)
{
    return ((math_fidelity_desc >> 3) & 0x1) + 1;
}

/**
 * @brief Calculates the maximum destination index for a matmul operation.
 *
 * Given the starting destination index and the dimensions ct_dim and rt_dim,
 * this function computes the maximum destination index accessed by the matmul kernel.
 * The addressing pattern always results in a maximum offset of ct_dim * rt_dim - 1.
 *
 * @param dst_index  Starting destination index
 * @param ct_dim     Column tile dimension (default 1)
 * @param rt_dim     Row tile dimension (default 1)
 * @return           Maximum destination index accessed (dst_index + ct_dim * rt_dim - 1)
 */
inline std::uint32_t get_dest_max_matmul_tiles(std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1)
{
    return dst_index + ct_dim * rt_dim - 1;
}

} // namespace ckernel::math
