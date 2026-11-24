// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h. Same instruction sequence and
// access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/move.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod
#include "tensor_shape.h"

using namespace ckernel;

namespace hal_port::eltwise_binary_custom
{
namespace rwc = hal::counters::rwc;

// TT_OP_ELW{MUL,SUB}(CLR_NONE, dest_accum, SRCB_BCAST_COL, addr_mod, 0)
template <EltwiseBinaryType eltwise_binary_type, std::uint8_t addr_mod, hal::fpu::DestWriteMode dest_write = hal::fpu::DestWriteMode::Overwrite>
inline constexpr hal::fpu::Elementwise bcast_col_op {
    .operation        = eltwise_binary_type == EltwiseBinaryType::ELWMUL ? hal::fpu::ElementwiseOperation::Multiply : hal::fpu::ElementwiseOperation::Subtract,
    .broadcast        = hal::fpu::SrcBBroadcast::Column,
    .dest_write       = dest_write,
    .address_modifier = addr_mod,
    .dest_row_offset  = 0,
    .release          = hal::fpu::SourceRelease::None};

// TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_AB)
inline constexpr rwc::MathCounter SET_AB {.src_a = rwc::Set(0), .src_b = rwc::Set(0)};
// TTI_SETRWC(CLR_A, 0, 0, 0, 0, SET_ABD_F)
inline constexpr rwc::MathCounter CLR_A_SET_ABD_F {
    .src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A, .clear_fidelity = true};
// TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_AB)
inline constexpr rwc::MathCounter CLR_B_SET_AB {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::B};
// TTI_SETRWC(CLR_A, 0, 0, 0, 0, SET_ABD)
inline constexpr rwc::MathCounter CLR_A_SET_ABD {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A};
// TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD)
inline constexpr rwc::MathCounter CLR_B_SET_ABD {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B};

// MOVB2D(0, src_row, ADDR_MOD_4, MOV_4_ROWS_D0_BRCST, dst_row). The rows are arguments of production's helper, so the
// word is encoded at compile time with both rows 0 and the rows added, as production's TTI_MOVB2D macro does.
inline constexpr hal::move::Transfer<hal::move::SrcB, hal::move::Dst> movb2d_col0_4rows(const std::uint32_t src_row, const std::uint32_t dst_row)
{
    return {
        .source_row      = src_row,
        .destination_row = dst_row,
        .address_mode    = ADDR_MOD_4,
        .number_of_rows  = hal::move::rows::Four,
        .broadcast       = hal::move::Broadcast::Column0};
}

inline constexpr std::uint32_t MOVB2D_COL0_4ROWS_BASE = movb2d_col0_4rows(0, 0).get_operation();
inline constexpr std::uint32_t MOVB2D_SRC_ROW_SHIFT   = 17;
static_assert(movb2d_col0_4rows(1, 0).get_operation() == MOVB2D_COL0_4ROWS_BASE + (1u << MOVB2D_SRC_ROW_SHIFT), "SrcB row field");
static_assert(movb2d_col0_4rows(60, 0).get_operation() == MOVB2D_COL0_4ROWS_BASE + (60u << MOVB2D_SRC_ROW_SHIFT), "SrcB row field");
static_assert(movb2d_col0_4rows(0, 1).get_operation() == MOVB2D_COL0_4ROWS_BASE + 1, "Dest row field");
static_assert(movb2d_col0_4rows(0, 60).get_operation() == MOVB2D_COL0_4ROWS_BASE + 60, "Dest row field");

} // namespace hal_port::eltwise_binary_custom

template <BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod_custom()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    hal_port::set_math_addr_mod<hal::cfg::Sec::S7, hal_port::MathAddrMod {.srca_incr = 8, .srcb_incr = srcb_incr, .dest_incr = 8}>(); // ADDR_MOD_7
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4> (ELWMUL only)
 * @param num_faces: Number of faces to process (1, 2, or 4)
 * @note Establishes the operand-driven default Src zero-substitution state, which the bcast-col compute asserts
 *       under LLK asserts; an op that leaves it at keep must be followed by this init again.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_init_custom_(const std::uint32_t num_faces)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod_custom<src_b_bcast_type>();

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::ClrDvalid::SrcA_Disable, hal::cfg::Sec::S0, 0>(), hal::cfg::set<hal::cfg::ClrDvalid::SrcB_Disable, hal::cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)

    // A preceding datacopy leaves the Src zero flag at keep; the FPU op needs the operand-driven value.
    hal_port::_configure_default_zero_flag_state_();
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

/**
 * @brief Emit one bcast-col FPU op (ELWMUL or ELWSUB) advancing srcA/srcB/dest per addr_mod.
 *
 * The op is fixed at compile time, so the constexpr branch resolves to a single emitted instruction.
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWSUB/ELWMUL>
 * @tparam addr_mod: addr-mod slot selecting the srcA/srcB/dest strides for this op.
 */
template <EltwiseBinaryType eltwise_binary_type, std::uint8_t addr_mod>
inline void _bcast_cols_op_()
{
    hal::fpu::run<hal_port::eltwise_binary_custom::bcast_col_op<eltwise_binary_type, addr_mod>>();
}

namespace detail
{
// One face-row of the bcast-col walk: four ops, two per face. The last op picks ADDR_MOD_4 (rewind to the tile
// start and step the fidelity phase) on the last face-row of a non-final phase, ADDR_MOD_6 otherwise.
template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_phases, std::uint32_t num_face_rows, std::size_t step>
inline void bcast_cols_face_row()
{
    constexpr std::size_t phase    = step / num_face_rows;
    constexpr std::size_t face_row = step % num_face_rows;
    constexpr bool rewind          = phase + 1 < num_phases && face_row + 1 == num_face_rows;

    // Even face (F0 / F2): op then rewind srcB by one face (ADDR_MOD_5) so the odd face reads it.
    _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_7>(); // srcB: 0 -> 8
    _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_5>(); // srcB: 8 -> 0

    // Odd face (F1 / F3): op then advance srcB +24 (ADDR_MOD_6) to the next face-row / tile,
    // or, on the last face-row of a non-final fidelity phase, rewind to the tile start and
    // step the phase (ADDR_MOD_4) so the next walk re-reads the same srcA/srcB block.
    _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_7>();                       // srcB: 0 -> 8
    _bcast_cols_op_<eltwise_binary_type, rewind ? ADDR_MOD_4 : ADDR_MOD_6>(); // srcB: 8 -> 32, or all -> 0, phase + 1
}

// One tile's ops for every fidelity phase and face-row, spelled out at compile time as a straight run of
// num_phases * num_face_rows * 4 immediates (no RISC loop between pushes).
template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_phases, std::uint32_t num_face_rows, std::size_t... Steps>
inline void bcast_cols_tile(std::index_sequence<Steps...>)
{
    (bcast_cols_face_row<eltwise_binary_type, num_phases, num_face_rows, Steps>(), ...);
}
} // namespace detail

/**
 * @brief Blocked bcast-col FPU scaffold shared by the SDPA SUB and indexer_score MUL paths.
 *
 * Single-sources the addr-mod setup, srcB face-reuse addressing, and per-tile counter resets; the two
 * paths differ only in the FPU op (selected at compile time). The name tracks the shared mechanism
 * (srcB face reuse), matching WH's _llk_math_sub_bcast_cols_reuse_custom_.
 *
 * The MUL path is a dest-MAC head reduction: ELWMUL's dest_accum_en field is 0 (MAC, not overwrite) and
 * dest is cleared only by tile_regs_acquire's ZEROACC, so the caller does ONE acquire per ct_dim-column
 * batch and calls once per head into the SAME dest -> dest[col] = sum_h qk[col,h]*w[h], one pack per
 * column instead of per (column, head). cb_qk must be head-major.
 *
 * Each 16x16 face pair (one face-row) is processed by four ops (two per face; each covers an aligned
 * 8x16 block). A full 32x32 tile has two face-rows (F0/F1 and F2/F3); a 16x32 tiny tile has a single
 * face-row (F0/F1).
 *
 * The MUL path honours math_fidelity like the canonical ELWMUL MOP: each tile is walked once per fidelity
 * phase, the last op of a non-final phase rewinds srcA/srcB/dest to the tile start and advances the
 * fidelity phase (ADDR_MOD_4), and the tile-end SETRWC resets the phase along with the counters. ELWMUL
 * accumulates into dest, so the phases sum in place exactly as in the canonical path.
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWSUB/ELWMUL>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4> (ELWMUL only)
 * @param ct_dim: Number of srcA column tiles processed in the block.
 * @param tensor_shape: Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
 * @param dst_index: Absolute dest tile slot where this block-row's ct_dim tiles begin. Multi-tile-row
 *                   callers (e.g. the fuser LoopBlockRow driver) advance this per block-row so each row
 *                   lands on its own dest slots; single-tile-row callers leave it at 0.
 * @note Canonical description of the shared blocked bcast-col mechanism; other files reference this one.
 * @note Call @ref _llk_math_eltwise_binary_init_custom_ first; re-run it after any op that leaves the Src zero flag
 *       at keep.
 */
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWMUL || eltwise_binary_type == EltwiseBinaryType::ELWSUB,
        "blocked bcast-col reuse scaffold supports ELWMUL and ELWSUB only");
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "bcast-col custom: Src zero-substitution flag does not hold the operand-driven value; an op between the init and "
        "this call left it at keep, so denormal Src operands would not be flushed");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Two faces make up one face-row; a full tile has two of them, a tiny tile one.
    const std::uint32_t num_face_rows = tensor_shape.num_faces_r_dim;
    LLK_ASSERT(num_face_rows == 1 || num_face_rows == 2, "bcast-col custom: one or two face-rows per tile");

    namespace cfg = hal::cfg;
    namespace ebc = hal_port::eltwise_binary_custom;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;

    set_math_addr_mod<cfg::Sec::S7, MathAddrMod {.srca_incr = 8, .srcb_incr = 8, .dest_incr = 8}>();         // ADDR_MOD_7
    set_math_addr_mod<cfg::Sec::S6, MathAddrMod {.srca_incr = 8, .srcb_incr = 24, .dest_incr = 8}>();        // ADDR_MOD_6
    set_math_addr_mod<cfg::Sec::S5, MathAddrMod {.srca_incr = 8, .srcb_incr = 0x3F & -8, .dest_incr = 8}>(); // ADDR_MOD_5: decrement srcB by 8

    // Fidelity phases (ELWMUL only): one walk of the tile per phase. The number of phases is the
    // MathFidelity value (HiFi2 = 2 ... HiFi4 = 4); LoFi is a single walk.
    constexpr std::uint32_t num_phases = is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1;
    if constexpr (num_phases > 1)
    {
        // Last op of a non-final phase: rewind srcA/srcB/dest to the tile start and step the fidelity phase.
        set_math_addr_mod<cfg::Sec::S4, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_clr = 1, .fid_incr = 1}>(); // ADDR_MOD_4
    }

    hal::counters::rwc::set<ebc::SET_AB>(); // reset both src counters to 0

    // Per tile: walk srcA/dest across 4 faces while srcB reuses one face per pair (F0/F1 -> srcB face
    // 0, F2/F3 -> srcB face 2). Comments are dest/srcA counter values. ADDR_MOD_5 rewinds srcB -8 so
    // the pair's 2nd op rereads the same srcB face; ADDR_MOD_6 then jumps srcB +24 to the next pair.
    // The face-row count is tested once per call, so each tile's ops go out as one straight run.
    auto walk_tiles = [&](auto face_rows)
    {
        constexpr std::uint32_t rows = decltype(face_rows)::value;
        for (std::uint32_t i = 0; i < ct_dim; i++)
        {
            // Position the dest write pointer at the start of this tile's 32x32 dest
            // slot (stride 64 rows). The packer indexes tiles at the full 32x32 tile
            // stride regardless of num_faces, so tiles must be written one-per-slot.
            // The ADDR_MOD dest increments below only advance 32 rows for a 16x32 tiny
            // tile (num_faces=2) and 64 rows for a full 32x32 tile (num_faces=4); the
            // per-slot base + counter reset makes both land on the right slot.
            hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + i);
            hal_port::clear_dst_reg_addr();

            detail::bcast_cols_tile<eltwise_binary_type, num_phases, rows>(std::make_index_sequence<num_phases * rows> {});

            // Rewind srcB to 0 for the next srcA tile (CLR_A clears srcA's dvalid); srcA keeps advancing.
            // SET_ABD_F also returns the fidelity phase to 0 after a HiFi walk (dest is re-based per tile).
            hal::counters::rwc::set<ebc::CLR_A_SET_ABD_F>();
        }
    };
    if (num_face_rows == 1)
    {
        walk_tiles(std::integral_constant<std::uint32_t, 1> {});
    }
    else
    {
        walk_tiles(std::integral_constant<std::uint32_t, 2> {});
    }

    // Clear srcB dvalid on the way out.
    hal::counters::rwc::set<ebc::CLR_B_SET_AB>();

    // Restore the dest base offset so the next op starts from tile slot 0.
    hal_port::clear_dst_reg_addr();
}

/**
 * @brief SDPA's SUB entry point into the shared blocked bcast-col scaffold.
 *
 * Kept as an op-named wrapper because external callers reference this name directly (WH arch + the
 * tt-llk eltwise_bcast_col tests), so it can't be folded away. The MUL path has no such external caller
 * and calls @ref _llk_math_bcast_cols_reuse_custom_ <ELWMUL> directly.
 *
 * @param ct_dim: Number of srcA column tiles processed in the block.
 * @param tensor_shape: Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
 * @param dst_index: Absolute dest tile slot where this block-row's ct_dim tiles begin.
 */
inline void _llk_math_sub_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    _llk_math_bcast_cols_reuse_custom_<EltwiseBinaryType::ELWSUB>(ct_dim, tensor_shape, dst_index);
}

/** @brief Initialises math state for cancellation-resistant column-broadcast subtraction. */
inline void _llk_math_sub_bcast_cols_compensated_init_()
{
    namespace cfg = hal::cfg;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    set_math_addr_mod<cfg::Sec::S3, MathAddrMod {.srca_incr = 8, .srcb_incr = 8, .dest_incr = 8}>();
    set_math_addr_mod<cfg::Sec::S5, MathAddrMod {.srca_incr = 8, .srcb_incr = 0x3F & -8, .dest_incr = 8}>();
    set_math_addr_mod<cfg::Sec::S6, MathAddrMod {.srca_incr = 8, .srcb_incr = 24, .dest_incr = 8}>();
    set_math_addr_mod<cfg::Sec::S4, MathAddrMod {}>();
    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::ClrDvalid::SrcA_Disable, hal::cfg::Sec::S0, 0>(), hal::cfg::set<hal::cfg::ClrDvalid::SrcB_Disable, hal::cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)
}

/**
 * @brief Broadcasts four SrcB column values into four rows of each destination face.
 * @param src_row First SrcB row containing the values to broadcast.
 * @param dst_row First destination row to write.
 */
inline void _compensated_move_broadcast_col_to_dest_(const std::uint32_t src_row, const std::uint32_t dst_row)
{
    // Blackhole forces MOVB2D's effective SrcA format to TF32 when FP32 DEST accumulation is
    // enabled. The move therefore writes 32-bit DEST rows without relying on implied source formats.
    namespace ebc = hal_port::eltwise_binary_custom;
    LLK_ASSERT(src_row + 12 + 4 <= 64 && dst_row + 12 + 4 <= 1024, "MOVB2D row out of range");
    TT_INSN(ebc::MOVB2D_COL0_4ROWS_BASE + ((src_row + 0) << ebc::MOVB2D_SRC_ROW_SHIFT) + (dst_row + 0));
    TT_INSN(ebc::MOVB2D_COL0_4ROWS_BASE + ((src_row + 4) << ebc::MOVB2D_SRC_ROW_SHIFT) + (dst_row + 4));
    TT_INSN(ebc::MOVB2D_COL0_4ROWS_BASE + ((src_row + 8) << ebc::MOVB2D_SRC_ROW_SHIFT) + (dst_row + 8));
    TT_INSN(ebc::MOVB2D_COL0_4ROWS_BASE + ((src_row + 12) << ebc::MOVB2D_SRC_ROW_SHIFT) + (dst_row + 12));
}

/**
 * @brief Computes (input - anchor) + (anchor - mean) for a block of 32x32 tiles.
 * @param ct_dim Number of consecutive input tiles to process.
 * @param tensor_shape Shape of each input tile; only full 32x32 tiles are supported.
 * @param dst_index First destination tile slot to write.
 * @note Call @ref _llk_math_sub_bcast_cols_compensated_init_ before this operation.
 * @note The input and anchor are consumed as TF32. Use an SFPU finaliser when FP32 mantissa preservation is required.
 */
inline void _llk_math_sub_bcast_cols_compensated_(
    const std::uint32_t ct_dim, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    LLK_ASSERT(tensor_shape.total_num_faces() == 4, "compensated column broadcast requires 32x32 tiles");

    namespace ebc = hal_port::eltwise_binary_custom;
    namespace rwc = hal::counters::rwc;
    using hal::fpu::DestWriteMode;
    hal_port::_configure_preserve_zero_flag_state_();
    for (std::uint32_t tile = 0; tile < ct_dim; ++tile)
    {
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + tile);
        hal_port::clear_dst_reg_addr();

        // Seed DEST with anchor - mean. The split statistic tile stores the
        // upper and lower row halves in faces 1 and 3 respectively.
        constexpr std::uint32_t upper_mean_row      = 16;
        constexpr std::uint32_t lower_mean_row      = 48;
        constexpr std::uint32_t upper_left_dst_row  = 0;
        constexpr std::uint32_t upper_right_dst_row = 16;
        constexpr std::uint32_t lower_left_dst_row  = 32;
        constexpr std::uint32_t lower_right_dst_row = 48;
        _compensated_move_broadcast_col_to_dest_(upper_mean_row, upper_left_dst_row);
        _compensated_move_broadcast_col_to_dest_(upper_mean_row, upper_right_dst_row);
        _compensated_move_broadcast_col_to_dest_(lower_mean_row, lower_left_dst_row);
        _compensated_move_broadcast_col_to_dest_(lower_mean_row, lower_right_dst_row);

        // Accumulate x - anchor into the correction already in DEST.
        for (std::uint32_t face_row = 0; face_row < tensor_shape.num_faces_r_dim; ++face_row)
        {
            hal::fpu::run<ebc::bcast_col_op<EltwiseBinaryType::ELWSUB, ADDR_MOD_3, DestWriteMode::Accumulate>>();
            hal::fpu::run<ebc::bcast_col_op<EltwiseBinaryType::ELWSUB, ADDR_MOD_5, DestWriteMode::Accumulate>>();
            hal::fpu::run<ebc::bcast_col_op<EltwiseBinaryType::ELWSUB, ADDR_MOD_3, DestWriteMode::Accumulate>>();
            hal::fpu::run<ebc::bcast_col_op<EltwiseBinaryType::ELWSUB, ADDR_MOD_6, DestWriteMode::Accumulate>>();
        }

        // Publish SrcA for the next input tile while keeping the split mean resident.
        rwc::set<ebc::CLR_A_SET_ABD>();
    }
    rwc::set<ebc::CLR_B_SET_ABD>();
    hal_port::_configure_default_zero_flag_state_();
    hal_port::clear_dst_reg_addr();
}
