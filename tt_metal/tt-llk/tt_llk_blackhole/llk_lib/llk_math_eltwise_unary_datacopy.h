// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h. Same Tensix instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/src.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

// local function declarations
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format);

namespace hal_port
{
namespace datacopy
{
namespace move = hal::move;

// The 32-bit broadcasts below issue MOVD2B/MOVB2D whose only runtime operand is the Dest row (dst_index * 64 + row).
// The rest of the word is encoded at compile time and the row added at runtime, as production's TT_OP_MOV* + dst does:
// the runtime move::Transfer path calls the non-inlined HAL encoder per instruction, which made GCC keep the whole
// per-tile datacopy out of line. The static_asserts check that the Dest row is the low field of the HAL encoding.

// MOVD2B(Lo32 ? DEST_32B_LOW : DEST_NORM, SrcBRow, ADDR_MOD_3, Rows == 4 ? MOV_4_ROWS : MOV_1_ROW, dest_row)
template <bool Lo32, std::uint32_t SrcBRow, std::uint16_t Rows>
constexpr std::uint32_t movd2b_operation(const std::uint32_t dest_row)
{
    return move::Transfer<move::Dst, move::SrcB> {
        .source_row = dest_row, .destination_row = SrcBRow, .address_mode = ADDR_MOD_3, .number_of_rows = move::RowCount {Rows}, .source_32bit_low = Lo32}
        .get_operation();
}

// MOVB2D(Lo32 ? DEST_32B_LOW : DEST_NORM, SrcBRow, ADDR_MOD_3, <Rows/Broadcast modifier>, dest_row):
// Row/8 = MOV_8_ROW_BRCST, Scalar/8 = MOV_8_ROW_BRCST_D0_BRCST, Column0/4 = MOV_4_ROWS_D0_BRCST.
template <bool Lo32, std::uint32_t SrcBRow, move::Broadcast B, std::uint16_t Rows>
constexpr std::uint32_t movb2d_operation(const std::uint32_t dest_row)
{
    return move::Transfer<move::SrcB, move::Dst> {
        .source_row            = SrcBRow,
        .destination_row       = dest_row,
        .address_mode          = ADDR_MOD_3,
        .number_of_rows        = move::RowCount {Rows},
        .broadcast             = B,
        .destination_32bit_low = Lo32}
        .get_operation();
}

// Dest rows the 32-bit broadcasts address: up to 15 tiles of 64 rows plus the in-tile row.
inline constexpr std::uint32_t MAX_BROADCAST_DEST_ROW = 15 * 64 + 63;

template <bool Lo32, std::uint32_t SrcBRow, std::uint16_t Rows>
inline __attribute__((always_inline)) void movd2b(const std::uint32_t dest_row)
{
    constexpr std::uint32_t base = movd2b_operation<Lo32, SrcBRow, Rows>(0);
    static_assert(
        movd2b_operation<Lo32, SrcBRow, Rows>(1) == base + 1 &&
        movd2b_operation<Lo32, SrcBRow, Rows>(MAX_BROADCAST_DEST_ROW + 1 - Rows) == base + MAX_BROADCAST_DEST_ROW + 1 - Rows);
    LLK_ASSERT(dest_row + Rows <= MAX_BROADCAST_DEST_ROW + 1, "MOVD2B Dest row out of range");
    TT_INSN(base + dest_row);
}

template <bool Lo32, std::uint32_t SrcBRow, move::Broadcast B, std::uint16_t Rows>
inline __attribute__((always_inline)) void movb2d(const std::uint32_t dest_row)
{
    constexpr std::uint32_t base = movb2d_operation<Lo32, SrcBRow, B, Rows>(0);
    static_assert(
        movb2d_operation<Lo32, SrcBRow, B, Rows>(1) == base + 1 &&
        movb2d_operation<Lo32, SrcBRow, B, Rows>(MAX_BROADCAST_DEST_ROW + 1 - Rows) == base + MAX_BROADCAST_DEST_ROW + 1 - Rows);
    LLK_ASSERT(dest_row + Rows <= MAX_BROADCAST_DEST_ROW + 1, "MOVB2D Dest row out of range");
    TT_INSN(base + dest_row);
}

// One hi16/lo16 pass of the ROW 32-bit broadcast: four MOVB2D(DEST_NORM) with SrcA=Tf32, then four
// MOVB2D(DEST_32B_LOW) with Fp32_enabled=0 and SrcA=Float32 (BH HW bug: MOVB2D DEST_32B_LOW is ignored while Fp32
// dest mode is on), then Fp32_enabled=1. R0..R3 are the dest rows each broadcast writes.
template <move::Broadcast B, std::uint32_t R0, std::uint32_t R1, std::uint32_t R2, std::uint32_t R3>
inline __attribute__((always_inline)) void broadcast_hi_lo_8rows(const std::uint32_t tile_base)
{
    // SrcA=Tf32: MOVB2D(DEST_NORM) writes hi16 of 32-bit dest
    set_srca_format<DataFormat::Tf32>();

    movb2d<false, p_movb2d::SRC_ROW16_OFFSET, B, 8>(tile_base + R0);
    movb2d<false, p_movb2d::SRC_ROW16_OFFSET, B, 8>(tile_base + R1);
    movb2d<false, p_movb2d::SRC_ROW16_OFFSET, B, 8>(tile_base + R2);
    movb2d<false, p_movb2d::SRC_ROW16_OFFSET, B, 8>(tile_base + R3);

    // BH HW bug: MOVB2D(DEST_32B_LOW) is ignored when Fp32 dest mode is active.
    set_fp32_enabled<0>();
    // SrcA=Float32 (non-Tf32): MOVB2D(DEST_32B_LOW) writes lo16 of 32-bit dest
    set_srca_format<DataFormat::Float32>();

    movb2d<true, p_movb2d::SRC_ZERO_OFFSET, B, 8>(tile_base + R0);
    movb2d<true, p_movb2d::SRC_ZERO_OFFSET, B, 8>(tile_base + R1);
    movb2d<true, p_movb2d::SRC_ZERO_OFFSET, B, 8>(tile_base + R2);
    movb2d<true, p_movb2d::SRC_ZERO_OFFSET, B, 8>(tile_base + R3);

    set_fp32_enabled<1>();
}

// TT_OP_SETRWC(CLR_AB, 0, 0, 0, 0, SET_AB)
inline constexpr std::uint32_t SETRWC_CLR_AB_SET_AB =
    rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}.operation();
// TT_OP_SETRWC(CLR_AB, 0, 0, 0, 0, 0)
inline constexpr std::uint32_t SETRWC_CLR_AB = rwc::MathCounter {.flip = rwc::Src::A | rwc::Src::B}.operation();
// TT_OP_SETRWC(CLR_B, CR_B, 0, 0, 0, SET_B)
inline constexpr std::uint32_t SETRWC_CLR_B_CR_B = rwc::MathCounter {.src_b = rwc::AdvanceAndReset(0), .flip = rwc::Src::B}.operation();
// HAL-GAP: MathCounter encodes a lone AdvanceAndReset (no flip) as INCRWC; nothing selects SETRWC(0, CR_B, 0, 0, 0, SET_B).
inline constexpr std::uint32_t SETRWC_CR_B_SET_B = TT_OP_SETRWC(0, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B);

// TT_OP_ELWADD(0, 0, broadcast, addr_mod, 0)
inline std::uint32_t elwadd_copy(const hal::fpu::SrcBBroadcast broadcast, const std::uint8_t addr_mod)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = hal::fpu::ElementwiseOperation::Add,
        .broadcast        = broadcast,
        .dest_write       = hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .dest_row_offset  = 0,
        .release          = hal::fpu::SourceRelease::None});
}

// TT_OP_MOVB2D(0, 0, addr_mod, <rows/broadcast>, 0)
inline std::uint32_t movb2d_copy(const move::RowCount rows, const move::Broadcast broadcast, const std::uint8_t addr_mod)
{
    return move::Transfer<move::SrcB, move::Dst> {
        .source_row = 0, .destination_row = 0, .address_mode = addr_mod, .number_of_rows = rows, .broadcast = broadcast}
        .get_operation();
}

// TT_OP_MOVA2D(Lo32 ? DEST_32B_LOW : DEST_NORM, 0, ADDR_MOD_2, MOV_8_ROWS, 0)
template <bool Lo32>
inline constexpr std::uint32_t MOVA2D_8_ROWS =
    move::Transfer<move::SrcA, move::Dst> {
        .source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_2, .number_of_rows = move::rows::Eight, .destination_32bit_low = Lo32}
        .get_operation();

} // namespace datacopy
} // namespace hal_port

/**
 * @brief Copy a tile into the destination register, optionally broadcasting source B.
 *
 * For the unpack-to-dest path with 32-bit data, applies the Blackhole hi16/lo16 broadcast workarounds
 * (Issue #449) and an extra zero-flag clear (budabackend#2730); otherwise runs the preconfigured datacopy MOP.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param dst_index: Tile index into the destination register.
 * @param src_format: Source data format (DataFormat enum underlying value).
 * @param dst_format: Destination data format (DataFormat enum underlying value).
 * @param tensor_shape: Tile geometry. Only the total face count is used (must be 1, 2, or 4). On 32-bit unpack to dest it
 *        sets how many faces get their Dest zero flags re-cleared (budabackend#2730), so it must cover at least the
 *        faces unpacked to Dest. @ref ckernel::DEFAULT_TENSOR_SHAPE (4 faces) is always valid.
 * @note Call @ref _llk_math_eltwise_unary_datacopy_init_ with matching template args before this
 *       function, and @ref _llk_math_eltwise_unary_datacopy_uninit_ after it to restore modified state.
 * @note On the unpack thread, @ref _llk_unpack_A_ must feed the tile into SrcA/SrcB (or dest for unpack-to-dest).
 */
template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_(
    const std::uint32_t dst_index,
    const std::uint32_t src_format,
    const std::uint32_t dst_format,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    namespace cfg = hal::cfg;
    namespace rwc = hal::counters::rwc;
    namespace dc  = hal_port::datacopy;
    namespace dst = hal::dst;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    using Bcast   = hal::move::Broadcast;

    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_unary_datacopy_", tensor_shape);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // For 32bit data, each half of DEST can take 16 tiles. Since dest offset is returned as if 16bit data are used, we need to
    // adjust it to offset in faces for 32bit data.
    if (unpack_to_dest && is_32bit_input(src_format, dst_format))
    {
        hal_port::math_unpack_to_dest_math_ready();
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        hal_port::math_unpack_to_dest_tile_ready();

        // Pin the math dest offset to the bank base: hardware adds it to the MOVD2B/MOVB2D immediates
        // (dst_index * 64 + row) of the broadcast sequences below, and a preceding op may have left another
        // tile's offset here. A plain copy (NONE) only issues the budabackend#2730 ZEROACC below, whose CLR_16
        // block index is absolute within the bank: the offset only feeds its bank select, which flips once
        // offset + index reaches 512 (tt-metal#53693), and a 32-bit bank never gets there (offset <= 240,
        // index <= 15). So NONE skips the write, as on Wormhole, and keeps its per-tile cost unchanged.
        if constexpr (src_b_bcast_type != BroadcastType::NONE)
        {
            // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base)
            cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(ckernel::get_dest_buffer_base());
        }

        // Due to bug in Blackhole Tensix (more details in budabackend/#2730) when an event with side effect of clearing DEST zero flags
        // (such as Unpack-to-dest or RISC-to-dest) and a ZEROACC instruction from packer occur in the same cycle,
        // zero flags clearing is dropped.
        // To mitigate that, we issue additional zero flag clear instruction immediately after unpack tile to dest is done.
        // RISC-to-dest event is not currently used.

        const std::uint32_t dst_format_masked = masked_data_format(dst_format);
        const int clear_fp32                  = static_cast<int>(
            dst_format_masked == static_cast<std::uint32_t>(DataFormat::Float32) || dst_format_masked == static_cast<std::uint32_t>(DataFormat::Int32) ||
            dst_format_masked == static_cast<std::uint32_t>(DataFormat::UInt32));
        const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < num_faces; i++)
        {
            // Clears zero flags in DEST for one face: TT_ZEROACC(CLR_16, clear_fp32, 1, ADDR_MOD_3, face).
            dst::run(dst::Zero {
                .scope        = dst::ZeroScope::Face,
                .index        = get_dest_index_in_faces(local_tile, i),
                .flags        = dst::ZeroFlagAction::ClearFlags,
                .width        = clear_fp32 ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16,
                .address_mode = ADDR_MOD_3});
        }

        if constexpr (src_b_bcast_type != BroadcastType::NONE)
        {
            // Disable implied SrcA format inference so manual SrcA format switches take effect.
            hal_port::set_disable_implied_srca_fmt<1>();
            hal_port::set_srca_format<DataFormat::Float32>();

            // Disable zero flag to prevent mantissa flushing when exponent bits are 0.
            cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 1>();

            // UNPACK_TO_DEST covers only unpacker 0 and MOVD2B doesn't wait for SrcB valid, so wait for
            // _llk_unpack_A_'s SrcB handoff (unpacker 1) before the broadcasts below write the bank.
            hal_port::srcb_bank_wait();
        }

        if constexpr (src_b_bcast_type == BroadcastType::ROW)
        {
            // Row broadcast for 32-bit unpack-to-dest: replicate one source row across all 16
            // rows of each face pair. 32-bit dest values are written in two passes (hi16 then lo16)
            // because MOVB2D can only write 16 bits at a time.
            //
            // The sequence per source row is:
            //   1. MOVD2B(DEST_NORM):    copy hi16 of source row from dest into B register
            //   2. MOVD2B(DEST_32B_LOW): copy lo16 of source row from dest into B register
            //   3. SrcA=Tf32 + MOVB2D(DEST_NORM, MOV_8_ROW_BRCST): broadcast hi16 from B to dest
            //   4. Fp32_enabled=0 (BH HW bug: MOVB2D DEST_32B_LOW ignores writes when Fp32 is on)
            //      SrcA=Float32 + MOVB2D(DEST_32B_LOW, MOV_8_ROW_BRCST): broadcast lo16 from B to dest
            //   5. Fp32_enabled=1: restore FP32 dest mode
            // _llk_unpack_A_ hands this thread the SrcB bank (SETDVALID is unsupported on Blackhole)
            const std::uint32_t tile_base = dst_index * 64;

            // Source row 0 -> faces 0 (rows 0-15) and 2 (rows 32-47)
            dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET, 1>(tile_base + 0); // hi16 to B
            dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET, 1>(tile_base + 0);   // lo16 to B
            dc::broadcast_hi_lo_8rows<Bcast::Row, 0, 8, 32, 40>(tile_base);

            // Source row 16 -> faces 1 (rows 16-31) and 3 (rows 48-63)
            dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET, 1>(tile_base + 16); // hi16 to B
            dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET, 1>(tile_base + 16);   // lo16 to B
            dc::broadcast_hi_lo_8rows<Bcast::Row, 16, 24, 48, 56>(tile_base);

            hal::src<hal::SrcB>.release(); // TTI_CLEARDVALID(0b10, 0)
        }
        else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
        {
            // Scalar broadcast for 32-bit unpack-to-dest: replicate datum[0] of row 0 across
            // every datum in every row of all 4 faces. Same hi16/lo16 split as ROW broadcast,
            // but uses MOV_8_ROW_BRCST_D0_BRCST which broadcasts both across rows AND across
            // all 16 datums within each row.
            // _llk_unpack_A_ hands this thread the SrcB bank (SETDVALID is unsupported on Blackhole)
            const std::uint32_t tile_base = dst_index * 64;

            // Extract source scalar from dest row 0 into B register
            dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET, 1>(tile_base + 0); // hi16 to B
            dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET, 1>(tile_base + 0);   // lo16 to B

            // SrcA=Tf32: MOVB2D(DEST_NORM) writes hi16 of 32-bit dest
            hal_port::set_srca_format<DataFormat::Tf32>();

            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 0);  // face 0 rows 0-7
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 8);  // face 0 rows 8-15
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 16); // face 1 rows 0-7
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 24); // face 1 rows 8-15
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 32); // face 2 rows 0-7
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 40); // face 2 rows 8-15
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 48); // face 3 rows 0-7
            dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Scalar, 8>(tile_base + 56); // face 3 rows 8-15

            // BH HW bug: MOVB2D(DEST_32B_LOW) is ignored when Fp32 dest mode is active.
            hal_port::set_fp32_enabled<0>();
            // SrcA=Float32 (non-Tf32): MOVB2D(DEST_32B_LOW) writes lo16 of 32-bit dest
            hal_port::set_srca_format<DataFormat::Float32>();

            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 0);  // face 0 rows 0-7
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 8);  // face 0 rows 8-15
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 16); // face 1 rows 0-7
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 24); // face 1 rows 8-15
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 32); // face 2 rows 0-7
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 40); // face 2 rows 8-15
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 48); // face 3 rows 0-7
            dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Scalar, 8>(tile_base + 56); // face 3 rows 8-15

            hal_port::set_fp32_enabled<1>();

            hal::src<hal::SrcB>.release(); // TTI_CLEARDVALID(0b10, 0)
        }
        else if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // Column broadcast for 32-bit unpack-to-dest: each source row's datum[0] is
            // replicated across all 16 datums of that row, preserving per-row uniqueness.
            // Processes 2 face pairs (offset 0: faces 0+1, offset 1: faces 2+3). Within
            // each pair, all 16 source rows are extracted in groups of 4, then broadcast
            // via MOV_4_ROWS_D0_BRCST to both the left face (rows 0-15) and the right
            // face (rows 16-31). Same hi16/lo16 split as ROW/SCALAR broadcasts.
            // _llk_unpack_A_ hands this thread the SrcB bank (SETDVALID is unsupported on Blackhole)
            const std::uint32_t tile_base = dst_index * 64;

#pragma GCC unroll 2
            for (int offset = 0; offset < 2; ++offset)
            {
                const std::uint32_t face_base = tile_base + offset * 32;

                // Extract all 16 source rows from dest into B register (4 rows at a time)
                dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET, 4>(face_base + 0);       // hi16 rows 0-3
                dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET + 4, 4>(face_base + 4);   // hi16 rows 4-7
                dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET + 8, 4>(face_base + 8);   // hi16 rows 8-11
                dc::movd2b<false, p_movd2b::SRC_ROW16_OFFSET + 12, 4>(face_base + 12); // hi16 rows 12-15
                dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET, 4>(face_base + 0);         // lo16 rows 0-3
                dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET + 4, 4>(face_base + 4);     // lo16 rows 4-7
                dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET + 8, 4>(face_base + 8);     // lo16 rows 8-11
                dc::movd2b<true, p_movd2b::SRC_ZERO_OFFSET + 12, 4>(face_base + 12);   // lo16 rows 12-15

                // SrcA=Tf32: MOVB2D(DEST_NORM) writes hi16 of 32-bit dest
                hal_port::set_srca_format<DataFormat::Tf32>();

                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Column0, 4>(face_base + 0);       // left face rows 0-3
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 4, Bcast::Column0, 4>(face_base + 4);   // left face rows 4-7
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 8, Bcast::Column0, 4>(face_base + 8);   // left face rows 8-11
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 12, Bcast::Column0, 4>(face_base + 12); // left face rows 12-15
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET, Bcast::Column0, 4>(face_base + 16);      // right face rows 0-3
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 4, Bcast::Column0, 4>(face_base + 20);  // right face rows 4-7
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 8, Bcast::Column0, 4>(face_base + 24);  // right face rows 8-11
                dc::movb2d<false, p_movb2d::SRC_ROW16_OFFSET + 12, Bcast::Column0, 4>(face_base + 28); // right face rows 12-15

                // BH HW bug: MOVB2D(DEST_32B_LOW) is ignored when Fp32 dest mode is active.
                hal_port::set_fp32_enabled<0>();
                // SrcA=Float32 (non-Tf32): MOVB2D(DEST_32B_LOW) writes lo16 of 32-bit dest
                hal_port::set_srca_format<DataFormat::Float32>();

                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Column0, 4>(face_base + 0);       // left face rows 0-3
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 4, Bcast::Column0, 4>(face_base + 4);   // left face rows 4-7
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 8, Bcast::Column0, 4>(face_base + 8);   // left face rows 8-11
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 12, Bcast::Column0, 4>(face_base + 12); // left face rows 12-15
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET, Bcast::Column0, 4>(face_base + 16);      // right face rows 0-3
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 4, Bcast::Column0, 4>(face_base + 20);  // right face rows 4-7
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 8, Bcast::Column0, 4>(face_base + 24);  // right face rows 8-11
                dc::movb2d<true, p_movb2d::SRC_ZERO_OFFSET + 12, Bcast::Column0, 4>(face_base + 28); // right face rows 12-15

                hal_port::set_fp32_enabled<1>();
            }

            hal::src<hal::SrcB>.release(); // TTI_CLEARDVALID(0b10, 0)
        }

        if constexpr (src_b_bcast_type != BroadcastType::NONE)
        {
            hal_port::set_disable_implied_srca_fmt<0>();

            // The 32b path manipulated the flag directly above (tt-llk#449 Fp32_enabled dance); invalidate the
            // tracked state so the next op re-applies the Src zero-substitution flag.
            math::_invalidate_src_zero_flag_state_();
        }
    }
    else
    {
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (is_fp32_dest_acc_en && src_b_bcast_type != BroadcastType::NONE)
        {
            // UInt16 case needs to use format switching for 32bit dest
            // without the debug bit 11 hack to write into high bits
            // avoiding BroadcastType::NONE mode as that path is used by SFPU
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                hal_port::set_disable_implied_srca_fmt<1>();
                cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 1>();
                hal_port::set_srca_format<DataFormat::Tf32>();
            }
        }

        if constexpr (type == DataCopyType::A2D)
        {
            Mop::run();
        }
        else if constexpr (type == DataCopyType::B2D)
        {
            if constexpr (src_b_bcast_type == BroadcastType::COL)
            {
                // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
                Mop::run();
                rwc::set<rwc::MathCounter {.flip = rwc::Src::B}>(); // TTI_SETRWC(CLR_B, 0, 0, 0, 0, 0)
                Mop::run();
                rwc::set<rwc::MathCounter {.flip = rwc::Src::A | rwc::Src::B}>(); // TTI_SETRWC(CLR_AB, 0, 0, 0, 0, 0)
            }
            else
            {
                Mop::run();
            }
        }

        if constexpr (is_fp32_dest_acc_en && src_b_bcast_type != BroadcastType::NONE)
        {
            // Undo format switching option
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                hal_port::set_disable_implied_srca_fmt<0>();
                math::_invalidate_src_zero_flag_state_();
            }
        }

        hal_port::clear_dst_reg_addr();
    }
}

/**
 * @brief Program the address-mod slots for a datacopy: single-row and 8-row (or 4-row for UInt16) dest/source steps.
 *
 * The increment pattern depends on the datacopy direction (A2D walks SrcA, B2D walks SrcB) and broadcast type;
 * UInt16 B2D uses 4-row steps because it relies on MOVB2D.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param dst_format: Destination data format (DataFormat enum underlying value); selects the UInt16 4-row step.
 */
template <DataCopyType type, BroadcastType bcast_type = BroadcastType::NONE>
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format)
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    set_math_addr_mod<Sec::S3, MathAddrMod {}>(); // ADDR_MOD_3: no increments

    // Use srcA for data movement
    if constexpr (type == DataCopyType::A2D) // Overrides BCAST_TYPE when type is A2D
    {
        set_math_addr_mod<Sec::S0, MathAddrMod {.srca_incr = 1, .dest_incr = 1}>();

        // Just unpack into A and move to Dest
        set_math_addr_mod<Sec::S2, MathAddrMod {.srca_incr = 8, .dest_incr = 8}>();
    }
    else
    {
        if constexpr (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR)
        {
            set_math_addr_mod<Sec::S0, MathAddrMod {.dest_incr = 1}>();

            // Just unpack into B and move to Dest
            set_math_addr_mod<Sec::S2, MathAddrMod {.dest_incr = 8}>();
        }
        else
        {
            set_math_addr_mod<Sec::S0, MathAddrMod {.srcb_incr = 1, .dest_incr = 1}>();

            // Just unpack into B and move to Dest
            if (dst_format == to_underlying(DataFormat::UInt16)) // UInt16 case needs to use MOVB2D, which is 4 rows per op
            {
                set_math_addr_mod<Sec::S2, MathAddrMod {.srcb_incr = 4, .dest_incr = 4}>();
            }
            else
            {
                set_math_addr_mod<Sec::S2, MathAddrMod {.srcb_incr = 8, .dest_incr = 8}>();
            }
        }
    }
}

/**
 * @brief Program the datacopy MOP, selecting the move instruction (MOVA2D/MOVB2D/ELWADD) per direction, format, and broadcast.
 *
 * A2D normally uses MOVA2D but falls back to ELWADD when dest is FP32/INT (except UInt16, which stays on MOVA2D) or the datum is UInt8; B2D uses
 * MOVB2D (or ELWADD for non-UInt16 column broadcast) with loop counts derived from the broadcast type.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam tilize: Pack in tilize layout (A2D only); collapses the outer loop to a single iteration.
 * @tparam is_int_fpu_en: Enable integer FPU datapath (forces the ELWADD move path like FP32 dest).
 * @param rows_per_inst: Rows moved per instruction (selects single-row vs. multi-row addr mode and loop count).
 * @param total_rows: Total rows to move across the inner loop.
 * @param tensor_shape: Tile geometry; only the total face count is used today.
 * @param dst_format: Destination data format (DataFormat enum underlying value); selects UInt16 special-casing.
 */
template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType bcast_type = BroadcastType::NONE, bool tilize = false, bool is_int_fpu_en = false>
inline void eltwise_unary_configure_mop(
    std::uint32_t rows_per_inst, std::uint32_t total_rows, const ckernel::TensorShape tensor_shape, const std::uint32_t dst_format)
{
    namespace mop  = hal::mop;
    namespace move = hal::move;
    namespace dc   = hal_port::datacopy;
    using hal::fpu::SrcBBroadcast;

    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_unary_configure_mop", tensor_shape);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    if constexpr (type == DataCopyType::A2D)
    {
        std::uint32_t innerloop = (rows_per_inst == p_mova2d::MOV_1_ROW) ? total_rows : (total_rows >> 3);
        std::uint32_t outerloop = tilize ? 1 : num_faces;

        if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) || (dst_format == to_underlying(DataFormat::UInt8)))
        {
            // use elwadd to handle unpacking data into src A as fp16, but dest is in fp32 mode OR to handle uint8 datums
            mop::program(mop::loop(outerloop, mop::loop(innerloop, dc::elwadd_copy(SrcBBroadcast::None, ADDR_MOD_2)), dc::SETRWC_CLR_AB_SET_AB));
        }
        else if (is_fp32_dest_acc_en && (dst_format == to_underlying(DataFormat::UInt16)))
        {
            // Typecasting uint16 to 32bit data, need data to be written to lower 16 bits without modification
            // to be consumed by SFPU easily.
            mop::program(mop::loop(outerloop, mop::loop(innerloop, dc::MOVA2D_8_ROWS<true>), dc::SETRWC_CLR_AB_SET_AB));
        }
        else
        {
            mop::program(mop::loop(outerloop, mop::loop(innerloop, dc::MOVA2D_8_ROWS<false>), dc::SETRWC_CLR_AB_SET_AB));
        }
    }
    else if constexpr (type == DataCopyType::B2D)
    {
        const std::uint8_t addr_mod = (rows_per_inst == p_movb2d::MOV_1_ROW) ? ADDR_MOD_0 : ADDR_MOD_2;
        std::uint32_t innerloop     = (rows_per_inst == p_movb2d::MOV_1_ROW) ? total_rows : (total_rows >> 2);
        // One outer iteration per face: end_op clears a SrcB DVALID each iteration.
        std::uint32_t outerloop = num_faces;

        if constexpr (bcast_type == BroadcastType::COL)
        {
            innerloop = 16 >> 3; // elwadd produces 8 rows per op
            // The mop only runs for 2 outer loops and mop is called twice for col broadcast
            outerloop = 2;
            // ELWADD with zeros will be used for non UInt16 case, since it moves 8 rows per cycle
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                innerloop = 16 >> 2; // movb2d produces 4 rows per op
            }
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            innerloop = (total_rows >> 3);
        }
        else if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            outerloop = 1;
            innerloop = num_faces * (total_rows >> 3);
        }

        // ELWADD moves 8 rows/cycle (vs MOVB2D's 4) but routes through the FPU, which interprets
        // UInt16 bit patterns as floats and flushes values with exp=0. UInt16 falls back to MOVB2D.
        if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            // UInt16: MOVB2D(MOV_8_ROW_BRCST_D0_BRCST); otherwise ELWADD(SRCB_BCAST_ALL)
            const std::uint32_t copy_op = (dst_format == to_underlying(DataFormat::UInt16))
                                              ? dc::movb2d_copy(move::rows::Eight, move::Broadcast::Scalar, addr_mod)
                                              : dc::elwadd_copy(SrcBBroadcast::Scalar, addr_mod);
            mop::program(mop::loop(outerloop, mop::loop(innerloop, copy_op), dc::SETRWC_CLR_AB));
        }
        else if constexpr (bcast_type == BroadcastType::COL)
        {
            // UInt16: MOVB2D(MOV_4_ROWS_D0_BRCST); otherwise ELWADD(SRCB_BCAST_COL)
            const std::uint32_t copy_op = (dst_format == to_underlying(DataFormat::UInt16))
                                              ? dc::movb2d_copy(move::rows::Four, move::Broadcast::Column0, addr_mod)
                                              : dc::elwadd_copy(SrcBBroadcast::Column, addr_mod);
            mop::program(mop::loop(outerloop, mop::loop(innerloop, copy_op), dc::SETRWC_CR_B_SET_B));
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            // MOVB2D(MOV_8_ROW_BRCST)
            mop::program(mop::loop(outerloop, mop::loop(innerloop, dc::movb2d_copy(move::rows::Eight, move::Broadcast::Row, addr_mod)), dc::SETRWC_CLR_B_CR_B));
        }
        else
        {
            // MOVB2D(rows_per_inst): MOV_1_ROW or MOV_4_ROWS
            LLK_ASSERT(rows_per_inst == p_movb2d::MOV_1_ROW || rows_per_inst == p_movb2d::MOV_4_ROWS, "B2D datacopy moves 1 or 4 rows per instruction");
            const move::RowCount rows = (rows_per_inst == p_movb2d::MOV_4_ROWS) ? move::rows::Four : move::rows::One;
            mop::program(mop::loop(outerloop, mop::loop(innerloop, dc::movb2d_copy(rows, move::Broadcast::None, addr_mod)), dc::SETRWC_CLR_B_CR_B));
        }
    }
}

/**
 * @brief Initialize the math thread (address mods and MOP) for an elementwise unary datacopy.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @tparam pack_mode: Packing layout, values = <Default/Tilize>
 * @param tensor_shape: Tile geometry (face_r_dim, face grid). Total face count must be 1, 2, or 4.
 * @param dst_format: Destination data format (DataFormat enum underlying value); 255 means unset.
 * @note COL broadcast supports only a full 32x32 tile (16-row faces, 2x2 face grid).
 * @note On the unpack thread, pair with @ref _llk_unpack_A_init_ (copy/transpose) or @ref _llk_unpack_tilize_init_ (tilize),
 * which feed the tile.
 * @note @ref _llk_math_eltwise_unary_datacopy_ runs the configured op with matching template args.
 */
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en             = false,
    PackMode pack_mode             = PackMode::Default>
inline void _llk_math_eltwise_unary_datacopy_init_(
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_format = 255)
{
    namespace cfg = hal::cfg;

    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Tilize,
        "Blackhole _llk_math_eltwise_unary_datacopy_init_ supports only PackMode::Default and PackMode::Tilize");
    constexpr bool tilize = (pack_mode == PackMode::Tilize);
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_unary_datacopy_init_", tensor_shape);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    if constexpr (src_b_bcast_type == BroadcastType::COL)
    {
        // The COL MOP covers two faces and the execute runs it twice, so it needs a full 32x32 tile
        // (4 faces of 16 rows). A 1x32 tile was seen to hang on Wormhole.
        LLK_ASSERT(
            tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
                tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
            "Column broadcast datacopy supports only 32x32 tiles");
    }
    eltwise_unary_configure_addrmod<type, src_b_bcast_type>(dst_format);

    if constexpr (type == DataCopyType::A2D && src_b_bcast_type == BroadcastType::NONE)
    {
        // Tilize: math MOP iterates over all faces of the tile (num_faces * 16 rows).
        // Non-tilize: single 16-row face pair (MOV_8_ROWS runs twice under the hood).
        eltwise_unary_configure_mop<type, is_fp32_dest_acc_en, src_b_bcast_type, tilize, is_int_fpu_en>(
            p_mova2d::MOV_8_ROWS, tilize ? (num_faces * 16) : 16, tensor_shape, dst_format);
    }
    else if constexpr (type == DataCopyType::B2D)
    {
        eltwise_unary_configure_mop<type, false, src_b_bcast_type>(p_movb2d::MOV_4_ROWS, 16, tensor_shape, dst_format);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f();
}

/**
 * @brief Uninitialize after an elementwise unary datacopy, undoing init-time workarounds.
 *
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Whether unpack wrote directly to dest.
 * unpack-to-dest.
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_uninit_()
{
}
