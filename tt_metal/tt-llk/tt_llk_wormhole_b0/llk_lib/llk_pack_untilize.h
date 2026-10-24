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
 * @brief Face columns of a tile: faces side by side in one tile row.
 *
 * 2 for a 32-wide tile (32x32, 16x32), 1 for a single-face tile and for a narrow (16-wide) tile such as
 * 32x16, whose two faces are stacked vertically. Untilize packs each tile row as the rows of this many
 * faces concatenated.
 *
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param narrow_tile: True for a 16-wide tile (one face column).
 * @return Face columns per tile row (1 or 2).
 */
inline std::uint32_t _llk_pack_untilize_num_faces_c_dim_(const std::uint32_t num_faces, const bool narrow_tile)
{
    return (num_faces == 1 || narrow_tile) ? 1 : 2;
}

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
 * @param narrow_tile: True for a 16-wide tile (one face column): a two-face narrow tile (32x16) stacks its faces
 *        vertically, so each packer emits one row of its own face instead of a row of faces 0 and 1 concatenated.
 * @note @ref _llk_pack_untilize_configure_addrmod_ must have programmed the ADDR_MOD slots.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool narrow_tile = false)
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
            if (_llk_pack_untilize_num_faces_c_dim_(num_faces, narrow_tile) == 2)
            {
                // Two face columns: each packer packs row Y of faces 0 and 1 (Dest rows Y and Y + 16) as one output row.
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
                // One face column: each packer packs row Y of its own face (packer k sits on face k through the
                // narrow_tile Dest offsets of _llk_init_packer_dest_offset_registers_) as one output row.
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
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param narrow_tile: True for a 16-wide tile (one face column); must match the narrow_tile given to
 *        @ref _llk_pack_dest_init_ / @ref _llk_init_packer_dest_offset_registers_ and to @ref _llk_pack_untilize_.
 * @note On the math thread, @ref _llk_math_eltwise_unary_datacopy_ (A2D) populates the dest register this packer reads.
 * @note Pair with @ref _llk_pack_untilize_uninit_ after the matching @ref _llk_pack_untilize_ execute calls.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_init_(
    const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool narrow_tile = false)
{
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // A narrow (16-wide) tile is one face column: a single face, or two full-height faces stacked. Two short faces
    // (face_r_dim < FACE_R_DIM) are rejected: the face_r_dim < FACE_R_DIM MOP enables one packer for face_r_dim rows,
    // so face 1 would never be packed.
    LLK_ASSERT(
        !narrow_tile || num_faces == 1 || (num_faces == 2 && face_r_dim == FACE_R_DIM),
        "a narrow (16-wide) tile is a single face or two full-height faces stacked");

    _llk_pack_untilize_configure_addrmod_<diagonal, narrow_row>();

    _llk_pack_untilize_mop_config_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(face_r_dim, num_faces, narrow_tile);

    if (block_ct_dim != full_ct_dim)
    {
        // Per-row L1 advance (matches BH): 2 face columns of datums per tile for a 32-wide tile, 1 for a single-face
        // or narrow tile.
        const std::uint32_t output_addr_offset =
            SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * _llk_pack_untilize_num_faces_c_dim_(num_faces, narrow_tile) * FACE_C_DIM);
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
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set.
 * @tparam tile_dst_ct_offset: Compile-time column-tile offset into the destination register.
 * @param address: L1 destination base address for the block.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param tile_dst_rt_offset: Runtime row-tile offset into the destination register.
 * @param narrow_tile: True for a 16-wide tile (one face column); must match the init. Each packer then owns a whole
 *        face and walks all its FACE_R_DIM rows, instead of half the rows of a two-face-column tile row.
 * @note Call @ref _llk_pack_untilize_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_untilize_uninit_ once all untilize-pack calls are complete.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool diagonal                    = false,
    bool narrow_row                  = false,
    std::uint32_t row_num_datums     = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0>
inline void _llk_pack_untilize_(
    const std::uint32_t address,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim         = FACE_R_DIM,
    const std::uint32_t tile_dst_rt_offset = 0,
    const bool narrow_tile                 = false)
{
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");

    program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal, row_num_datums>(address, pack_dst_format);

    if constexpr (narrow_row)
    {
        ckernel::ckernel_template::run();
    }
    else
    {
        // Full-height faces: packers 0..3 read Dest rows Y + {0, 8, 32, 40} of a 32-wide tile, each PACR packing the
        // same row of faces 0+1 (or 2+3), so each packer walks half the rows of a face row (TILE_R_DIM / 4 runs); for a
        // narrow tile they sit on rows Y + {0, 16} and each walks its whole face (FACE_R_DIM runs). The diagonal layout keeps its row count (narrow tiles are
        // not laid out for it). program_packer_untilized_destination starts packer p at p * (TILE_R_DIM / 4) rows of full_ct_dim * TILE_C_DIM datums; for a
        // narrow tile that is p * FACE_R_DIM rows of full_ct_dim * FACE_C_DIM datums, the same byte count, so packer 1 starts at the first output row of face 1
        // without a narrow case there.
        const std::uint32_t num_rows = (face_r_dim < FACE_R_DIM) ? face_r_dim : ((!diagonal && narrow_tile) ? FACE_R_DIM : TILE_R_DIM / 4);

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
