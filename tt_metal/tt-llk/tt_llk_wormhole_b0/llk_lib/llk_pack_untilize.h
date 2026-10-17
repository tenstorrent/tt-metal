// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"
#include "lltt.h"
#include "sfpi.h"

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Configure the ADDR_MOD slots used by the untilize pack MOP.
 *
 * Programs the src Y increment/clear patterns the untilize MOP relies on to walk face rows and reset
 * between rows/faces for the selected layout.
 *
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam narrow_row: True when packing fewer than a full row of datums.
 */
template <bool diagonal = false, bool narrow_row = false>
inline void _llk_pack_untilize_configure_addrmod_()
{
    if constexpr (diagonal || narrow_row)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 1},
        }
            .set(ADDR_MOD_0);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        }
            .set(ADDR_MOD_0);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_2);

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

/**
 * @brief Build and program the packer MOP template for an untilize (tilized -> row-major) pack.
 *
 * Programs a MOP that walks face rows and tiles within the block, packing rows out in row-major
 * order, and (when packing a partial block) records a replay buffer that advances the L1 destination
 * address by the per-row stride.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam narrow_row: True when packing fewer than a full row of datums.
 * @tparam row_num_datums: Number of datums per output row.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note @ref _llk_pack_untilize_configure_addrmod_ must have programmed the ADDR_MOD slots.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t PACKCNT              = diagonal ? (num_faces > 2 ? num_faces / 2 : num_faces) : ((face_r_dim < FACE_R_DIM) ? 1 : num_faces);
    constexpr std::uint32_t MEGAROW          = 1;
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t MOP_INNER_LOOP   = narrow_row ? (TILE_R_DIM / 4) : (diagonal ? FACE_R_DIM - 1 : 1);
    constexpr std::uint32_t MOP_OUTER_LOOP   = narrow_row ? 1 : block_ct_dim;

    if constexpr (diagonal)
    {
        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_INCADCXY(p_setadc::PAC, 0, 1, 0, 1),
            TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_end_ops(TT_OP_SETADCXX(p_setadc::PAC, 1 - 1, 0x0), TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
        tmp.program();
    }
    else
    {
        if constexpr (narrow_row)
        { // always
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
            tmp.program();
        }
        else
        {
            if (num_faces > 1)
            {
                // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
                ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
                tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
                tmp.set_end_ops(
                    TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0),
                    TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
                tmp.program();
            }
            else
            {
                ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
                tmp.set_end_op(TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
                tmp.program();
            }
        }

        if (block_ct_dim != full_ct_dim)
        {
            const std::uint32_t replay_buf_len = 10;
            lltt::record(ckernel::packer::replay_buf_offset, replay_buf_len);
            TTI_PACR(ADDR_MOD_3, 0, 0xf, 0, 0, 1, 1); // close block
            // update l1 address
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 1, p_gpr_pack::OUTPUT_ADDR + 1, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 2, p_gpr_pack::OUTPUT_ADDR + 2, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 3, p_gpr_pack::OUTPUT_ADDR + 3, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 3);
            TTI_NOP;
        }
    }
}

/**
 * @brief Initialize the packer for an untilize pack op.
 *
 * Configures ADDR_MODs and the untilize MOP, and (when packing a partial block) stores the per-row L1
 * destination address offset so the MOP can advance the L1 address per row, then programs the packer
 * X counter for the row width.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks (must be divisible by block_ct_dim).
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam narrow_row: True when packing at most FACE_C_DIM datums per row (see row_num_datums).
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set, at most FACE_C_DIM.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note On the math thread, @ref _llk_math_eltwise_unary_datacopy_ (A2D) populates the dest register this packer reads.
 * @note Pair with @ref _llk_pack_untilize_uninit_ after the matching @ref _llk_pack_untilize_ execute calls.
 * @note narrow_row packs exactly one full-height four-face tile: block_ct_dim == full_ct_dim == 1, num_faces == 4,
 * face_r_dim == FACE_R_DIM. The narrow MOP is one pass of TILE_R_DIM / 4 PACRs with no tile advance, each selected
 * packer (PACKCNT == num_faces) takes eight rows from its dest offset (@ref _llk_init_packer_dest_offset_registers_:
 * packer 1 at face_r_offset, packers 2 and 3 at 32 and 32 + face_r_offset), and @ref _llk_pack_untilize_ neither
 * replays the row-advance buffer nor closes a partial block for it, so a wider block or several blocks leave rows
 * unpacked. A two-face tile is rejected because the init cannot tell its shape from num_faces: a 16x32 tile
 * (face_r_offset 8) would pack completely, but a 32x16 tile is a narrow tile with face_r_offset FACE_R_DIM, so packer
 * 1 starts at face 1 and rows 8-15 of each face are never packed; a short face misplaces the packers likewise.
 * Checked by static_assert (dims) and LLK_ASSERT (faces).
 * @note The packers' L1 start addresses are 16-byte granular (program_packer_untilized_destination places packer k
 * at k * row_num_datums * datum bytes * TILE_R_DIM / 4 bytes, floored to 16), so the eight rows a packer writes must
 * span a multiple of 16 bytes: with 1-byte formats row_num_datums must be even. Checked by LLK_ASSERT.
 * @note The narrow MOP never sets or advances the dest tile, so @ref _llk_pack_untilize_ packs dest tile 0 of one
 * row block only: tile_dst_ct_offset and tile_dst_rt_offset must be 0 (static_assert / LLK_ASSERT there) and the
 * wrapper's block_rt_dim must be 1 (LLK_ASSERT in llk_pack_untilize).
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_init_(const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    if constexpr (narrow_row)
    {
        // A narrow row is at most one face wide: the per-row PACR (SETADCXX below) reads row_num_datums consecutive
        // datums of one face, so a wider row reads past the face and is packed wrong (measured: 24 bf16 datums).
        // Zero datums would wrap the SETADCXX X-end (row_num_datums - 1), so the lower bound is explicit.
        static_assert(row_num_datums >= 1 && row_num_datums <= FACE_C_DIM, "row_num_datums must be between 1 and FACE_C_DIM for narrow_row packing");
        // The narrow MOP (see _llk_pack_untilize_mop_config_) is one pass of TILE_R_DIM / 4 PACRs with no tile
        // advance, and _llk_pack_untilize_ runs it once per call without the row-advance replay or a partial-block
        // close, so only a single tile per call is packed: one block of one tile.
        static_assert(block_ct_dim == 1 && full_ct_dim == 1, "narrow_row packing requires block_ct_dim == full_ct_dim == 1");
    }
    // Each PACR of the narrow MOP packs one row per selected packer (PACKCNT == num_faces) for eight rows from the
    // packer's dest offset. A two-face tile may be 32x16 (a narrow tile, face_r_offset FACE_R_DIM), which loses rows
    // 8-15 of each face, and the init cannot tell it from a 16x32 tile; a short face misplaces the packers likewise.
    LLK_ASSERT(!narrow_row || (num_faces == 4 && face_r_dim == FACE_R_DIM), "narrow_row packing requires four full-height faces");
    // program_packer_untilized_destination places packer k at (k * row_num_datums * block_size) / 16 / TILE_C_DIM
    // 16-byte units, i.e. k times the bytes of the eight rows a packer writes, floored to 16 bytes; a 1-byte format
    // with an odd row_num_datums (5 datums: packer 1 at 32 bytes while packer 0 writes 40) overlaps the packers.
    LLK_ASSERT(
        !narrow_row || (SCALE_DATUM_SIZE(pack_dst_format, (TILE_R_DIM / 4) * row_num_datums) % 16) == 0,
        "narrow_row packing requires the eight rows a packer writes to span a multiple of 16 bytes (packer start addresses are "
        "16-byte granular)");

    _llk_pack_untilize_configure_addrmod_<diagonal, narrow_row>();

    _llk_pack_untilize_mop_config_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(face_r_dim, num_faces);

    if (block_ct_dim != full_ct_dim)
    {
        // Match BH: 2 face-cols per tile-row for num_faces>=2; old WH `num_faces/2` gave 1 only at num_faces==2 (#52175).
        const std::uint32_t output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces == 1) ? 1 : 2) * FACE_C_DIM);
        TT_SETDMAREG(0, LOWER_HALFWORD(output_addr_offset / 16), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET)); // store 16B aligned row offset address
    }

    // Pack row by row
    if constexpr (diagonal)
    {
        TTI_SETADCXX(p_setadc::PAC, 0, 0x0);
    }
    else if constexpr (narrow_row)
    {
        TTI_SETADCXX(p_setadc::PAC, row_num_datums - 1, 0x0);
    }
    else
    {
        TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    }
}

/**
 * @brief No-op teardown after an untilize pack op.
 *
 * The packer x-start/x-end is transient and reprogrammed by each operation's init (see tt-llk#1036),
 * so there is nothing to restore here.
 *
 * @note Call @ref _llk_pack_untilize_init_ before this function.
 */
inline void _llk_pack_untilize_uninit_()
{
}

/**
 * @brief Untilize-pack one block of tiles from the destination register to L1.
 *
 * Programs the L1 destination address and the packer counters, then runs the MOP once per face group,
 * advancing counters between face groups and resetting them afterward.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam narrow_row: True when packing at most FACE_C_DIM datums per row (see row_num_datums).
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set, at most FACE_C_DIM.
 * @tparam tile_dst_ct_offset: Compile-time column-tile offset into the destination register (0 with narrow_row).
 * @param address: L1 destination base address for the block.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param tile_dst_rt_offset: Runtime row-tile offset into the destination register (0 with narrow_row).
 * @note Call @ref _llk_pack_untilize_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_untilize_uninit_ once all untilize-pack calls are complete.
 * @note With narrow_row the MOP is run once as programmed, without the per-row set_dst_write_addr of the full-row
 *       path, so it always packs dest tile 0: both dest offsets must be 0 (see the init's @note).
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool diagonal                    = false,
    bool narrow_row                  = false,
    std::uint32_t row_num_datums     = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0>
inline void _llk_pack_untilize_(
    const std::uint32_t address, const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_dst_rt_offset = 0)
{
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    // The narrow branch below runs the MOP without set_dst_write_addr and the narrow MOP has no dest-tile advance,
    // so a nonzero dest offset would be ignored and tile 0 packed in its place.
    static_assert(!narrow_row || tile_dst_ct_offset == 0, "narrow_row packing packs dest tile 0: tile_dst_ct_offset must be 0");
    LLK_ASSERT(!narrow_row || tile_dst_rt_offset == 0, "narrow_row packing packs dest tile 0: tile_dst_rt_offset must be 0");

    program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal, row_num_datums>(address, pack_dst_format);

    if constexpr (narrow_row)
    {
        ckernel::ckernel_template::run();
    }
    else
    {
        const std::uint32_t num_rows = (face_r_dim < FACE_R_DIM) ? face_r_dim : TILE_R_DIM / 4;

        for (std::uint32_t row = 0; row < num_rows; row++)
        {
            set_dst_write_addr(tile_dst_ct_offset + tile_dst_rt_offset); // Clear tile counter
            ckernel::ckernel_template::run();
            TTI_ADDRCRXY(p_setadc::PAC, 0, 0, 1, 0, 0b0010); // Read new row in the tile
            if constexpr (block_ct_dim != full_ct_dim)
            {
                lltt::replay(ckernel::packer::replay_buf_offset, 10); // update row address
            }
        }
    }

    if constexpr (block_ct_dim == full_ct_dim)
    {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close block
    }
}
