// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// #include "kernel_types.h"
#include "ckernel.h"
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
// Src zero-substitution flag (ALU_ACC_CTRL_Zero_Flag_disabled_src) state tracker.
//
// The flag is a math-ALU concern: it is only read by MOVA2D/MOVB2D/MOVB2A/MVMUL/ELWADD/ELWMUL (the
// math thread), and is NOT read by the SFPU. So the math thread owns it via a small state machine.
//
// Programming model: EVERY op establishes its own zero-flag policy in its init — no op inherits it,
// so the flag's value is always determined by the current op, never by execution history. There are
// exactly two policies (UNCONFIGURED is only the power-on sentinel that forces the first write):
//
//   PRESERVE : flag = 1 (keep). Data-movement ops (datacopy / copy_init / transpose_dest / reduce
//              mov-phase) pass values through faithfully. This preserves bf16 -0.0 (which the SFPU
//              sign ops signbit/copysign read back out of DEST) and the 16b/32b integer datums that
//              would otherwise be corrupted. See ckernel::requires_disabled_src_zero_flag.
//   DEFAULT  : flag follows the operand formats (keep for the int formats that require it, flush
//              otherwise). FP-compute ops (matmul / eltwise-binary / reduce compute-phase) and format
//              reconfigs use this; it is the characterized, shipped behavior for MVMUL / ELW.
//
// Each configurator early-returns when already in its policy (DEFAULT additionally re-applies when
// the cached operand formats changed), so steady-state ops pay no extra cfg writes. Any path that
// writes the flag directly (bypassing the tracker) MUST call _invalidate_src_zero_flag_state_() so
// the next configurator re-applies regardless of the skip-if-set fast path.
// (Blackhole reduce's enforce_fp32 path uses ALU_ACC_CTRL_Fp32_enabled for the same MOVD2B/B2D
// workaround - Issue tt-llk#449 - so it does not interact with this flag.)
// ---------------------------------------------------------------------------------------------
enum class SrcZeroFlagState : std::uint8_t
{
    UNCONFIGURED = 0,
    DEFAULT      = 1,
    PRESERVE     = 2,
};

static SrcZeroFlagState src_zero_flag_state = SrcZeroFlagState::UNCONFIGURED;
static std::uint32_t src_zero_flag_srca_fmt = 0xff;
static std::uint32_t src_zero_flag_srcb_fmt = 0xff;
// Last value physically written to ALU_ACC_CTRL_Zero_Flag_disabled_src (0xff = unknown). DEFAULT is
// re-established on every format-changing binary init (e.g. groupnorm's mul_bcast loop), but
// requires_disabled_src_zero_flag() maps all float formats to the same bit -- so those re-applies are
// no-ops at the HW level. Caching the written value lets the writer skip the pipe-draining
// STALLWAIT + RMW when the bit is unchanged (tt-metal#49924).
static std::uint32_t src_zero_flag_hw = 0xff;

inline void _configure_src_zero_flag_(const bool disable)
{
    const std::uint32_t next = disable ? 1 : 0;
    if (src_zero_flag_hw == next)
    {
        return;
    }
    src_zero_flag_hw = next;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(next);
}

// DEFAULT slow path: the actual state update + flag write. Kept out-of-line so the hot-loop fast path
// below stays call-free and so the STALLWAIT + RMW exist in a single copy (code size — a matmul kernel
// otherwise overflows its slot).
inline __attribute__((noinline)) void _apply_default_zero_flag_state_(const std::uint32_t srca_dst_format, const std::uint32_t srcb_dst_format)
{
    src_zero_flag_srca_fmt = srca_dst_format;
    src_zero_flag_srcb_fmt = srcb_dst_format;
    src_zero_flag_state    = SrcZeroFlagState::DEFAULT;
    // TODO(tt-metal#53652): per mcraigheadTT / ttsim#713 the flag must be CLEARED for all FPU compute;
    // once that lands this becomes _configure_src_zero_flag_(false) and requires_disabled_src_zero_flag()
    // is dropped entirely (it is only reachable from here).
    _configure_src_zero_flag_(requires_disabled_src_zero_flag(srca_dst_format, srcb_dst_format));
}

// DEFAULT: the flag follows the operand formats. The fast path is inlined at every init call site and
// returns when the flag already holds the required value while in DEFAULT -- the steady state inside hot
// loops (e.g. groupnorm's mul_bcast, whose operand formats change each iteration but all map to the same
// flag value). Gating on the resulting value rather than the formats keeps that loop call-free; only a
// genuine policy/value change pays the out-of-line _apply_. (tt-metal#49924.)
inline void _configure_default_zero_flag_state_(const std::uint32_t srca_dst_format, const std::uint32_t srcb_dst_format)
{
    // TODO(tt-metal#53652): when DEFAULT always clears, this fast path reduces to
    // (src_zero_flag_state == DEFAULT && src_zero_flag_hw == 0) -- no requires_disabled_src_zero_flag call.
    const std::uint32_t want = requires_disabled_src_zero_flag(srca_dst_format, srcb_dst_format) ? 1u : 0u;
    if (src_zero_flag_state == SrcZeroFlagState::DEFAULT && src_zero_flag_hw == want)
    {
        return;
    }
    _apply_default_zero_flag_state_(srca_dst_format, srcb_dst_format);
}

// PRESERVE: data-movement ops (datacopy / copy_init / transpose_dest / reduce mov-phase) keep the
// flag disabled so values pass through faithfully (preserves bf16 -0.0 and 16b/32b int datums).
inline void _configure_preserve_zero_flag_state_()
{
    if (src_zero_flag_state == SrcZeroFlagState::PRESERVE)
    {
        return;
    }
    src_zero_flag_state = SrcZeroFlagState::PRESERVE;
    _configure_src_zero_flag_(true);
}

// Invalidate the tracked state after a code path that writes the flag directly (bypassing the
// tracker), so the next configurator re-applies regardless of the skip-if-set fast path. The
// physical value is now unknown too, so drop the write-cache to force the next _configure_ to write.
inline void _invalidate_src_zero_flag_state_()
{
    src_zero_flag_state = SrcZeroFlagState::UNCONFIGURED;
    src_zero_flag_hw    = 0xff;
}

inline void reset_counters(const std::uint32_t setrwc)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
}

inline void incr_counters(const std::uint32_t incr_a, const std::uint32_t incr_b, const std::uint32_t incr_d, const std::uint32_t incr_cr)
{
    TT_INCRWC(incr_cr, incr_d, incr_b, incr_a);
}

inline void move_d2a_fixed_face(const std::uint8_t addrmod)
{
    // Drain preceding math instructions so their source-bank release is visible before testing
    // the DVALID of the bank that MOVD2A will write.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, addrmod, p_movd2a::MOV_4_ROWS, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, addrmod, p_movd2a::MOV_4_ROWS, 4);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, addrmod, p_movd2a::MOV_4_ROWS, 8);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, addrmod, p_movd2a::MOV_4_ROWS, 12);
}

inline void move_d2b_fixed_face(const std::uint8_t addrmod)
{
    // Drain preceding math instructions so their source-bank release is visible before testing
    // the DVALID of the bank that MOVD2B will write.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
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

inline constexpr bool is_32bit_input(const std::uint32_t src_format, const std::uint32_t dst_format)
{
    const std::uint32_t input_df  = masked_data_format(src_format);
    const std::uint32_t output_df = masked_data_format(dst_format);
    return ((input_df == to_underlying(DataFormat::Int32)) || (input_df == to_underlying(DataFormat::Float32))) &&
           ((output_df == to_underlying(DataFormat::Int32)) || (output_df == to_underlying(DataFormat::Float32)));
}

inline constexpr bool is_high_fidelity(const MathFidelity math_fidelity_desc)
{
    return math_fidelity_desc != MathFidelity::LoFi;
}

// Returns the offset represented in DEST rows for a given face of a given tile.
inline std::uint32_t get_dest_index_in_faces(const std::uint32_t dst_index, const std::uint32_t face_index)
{
    // dst_index << 2 gives a tile idex in faces, because there are 4 faces in a tile.
    // face_index should normally take values from {0, 1, 2, 3}. If it's greater
    // than 3 faces from next tiles would be accessed.
    LLK_ASSERT(face_index < 4, "face_index out of range");
    return (dst_index << 2) + face_index;
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
