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

/// Packer L1 addresses (the per-row stride GPR, a block start) are kept in units of this many bytes.
constexpr std::uint32_t PACK_UNTILIZE_L1_ADDRESS_UNIT_BYTES = 16;

/**
 * @brief Dest rows (MOP runs) an untilize pack walks per block.
 *
 * Every run packs one output row per enabled packer: two PACRs per packer for the multi-face full-row
 * MOP (row Y of faces 0 and 1, concatenated), one PACR for the narrow-row and single-face MOPs. In the
 * non-diagonal layout packers 0..3 read the physical Dest rows Y + {0, 8, 32, 40} (see
 * @ref _llk_init_packer_dest_offset_registers_), so a 32x32 tile is covered by 4 packers x 8 runs and a
 * 16-row two-face tile by packers 0/1 x 8 runs. A single-face tile (num_faces == 1) enables packer 0 only,
 * so it has to walk all face_r_dim rows itself; tiles with fewer than FACE_R_DIM rows also use one packer
 * and need face_r_dim runs.
 *
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @return Dest rows (MOP runs) per block.
 */
inline std::uint32_t _llk_pack_untilize_num_rows_(const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    return (face_r_dim < FACE_R_DIM) ? face_r_dim : ((num_faces == 1) ? FACE_R_DIM : TILE_R_DIM / 4);
}

/**
 * @brief True for a one-tile narrow block (block_ct_dim == full_ct_dim == 1 with narrow_row), which is packed
 *        by a single MOP run of @ref _llk_pack_untilize_num_rows_ PACRs instead of one run per row.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim, bool narrow_row>
inline constexpr bool _llk_pack_untilize_single_tile_v = narrow_row && block_ct_dim == 1 && full_ct_dim == 1;

/**
 * @brief Configure the ADDR_MOD slots used by the untilize pack MOP.
 *
 * Programs the src Y increment/clear patterns the untilize MOP relies on to walk face rows and reset
 * between rows/faces for the selected layout.
 *
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam step_y_per_pacr: True when ADDR_MOD_0 has to step Y by one row after every PACR (diagonal, and the
 *         one-tile narrow block whose single MOP run walks the rows itself).
 */
template <bool diagonal = false, bool step_y_per_pacr = false>
inline void _llk_pack_untilize_configure_addrmod_()
{
    if constexpr (diagonal || step_y_per_pacr)
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
 * @brief Program the one-PACR-per-tile untilize MOP.
 *
 * One run packs row Y of every tile in the block: each PACR uses ADDR_MOD_1, which reloads Y from its CR so
 * the next tile reads the same row, and INCADCZW steps W to the next tile. The caller advances the row
 * between runs. The PACR packs the X range set by SETADCXX in init: a full face row for a single-face tile,
 * the first row_num_datums datums of it for a narrow row.
 *
 * @tparam concat: PACR Concat field; only meaningful with compression (narrow rows keep 0, full rows MEGAROW).
 * @param outer_loop: MOP outer loop count (tiles per block).
 * @param inner_loop: MOP inner loop count (PACRs per tile, 1).
 * @param pack_sel: Packer mask.
 */
template <std::uint32_t concat>
inline void _llk_pack_untilize_program_row_mop_(const std::uint32_t outer_loop, const std::uint32_t inner_loop, const std::uint32_t pack_sel)
{
    ckernel::ckernel_template tmp(outer_loop, inner_loop, TT_OP_PACR(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, pack_sel, 0, concat, 0, 0));
    tmp.set_end_op(TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
    tmp.program();
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
 * @tparam row_num_datums: Datums each tile contributes to an output row (TILE_C_DIM unless narrow_row); an
 *         output row of a block holds block_ct_dim * row_num_datums datums.
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
    constexpr bool single_tile               = _llk_pack_untilize_single_tile_v<block_ct_dim, full_ct_dim, narrow_row>;
    const std::uint32_t MOP_INNER_LOOP       = single_tile ? _llk_pack_untilize_num_rows_(face_r_dim, num_faces) : (diagonal ? FACE_R_DIM - 1 : 1);
    constexpr std::uint32_t MOP_OUTER_LOOP   = single_tile ? 1 : block_ct_dim;

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
        if constexpr (single_tile)
        {
            // One-tile narrow block: a single MOP run of num_rows PACRs packs the whole tile, ADDR_MOD_0 stepping
            // Y by one row after each PACR (no W advance, no per-row counter traffic from the execute loop).
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
            tmp.program();
        }
        else if constexpr (narrow_row)
        {
            // Narrow row of every tile in the block; the row advances between MOP runs, so block_ct_dim tiles are
            // covered per row, the layout program_packer_untilized_destination lays the per-packer L1 bases out for.
            _llk_pack_untilize_program_row_mop_<0>(MOP_OUTER_LOOP, MOP_INNER_LOOP, PACK_SEL(PACKCNT));
        }
        else if (num_faces > 1)
        {
            // Two PACRs per tile, row Y of faces 0 and 1: inc ch0_y+=1 (addr_mod_0 will increment by 15)
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.set_end_ops(
                TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0),
                TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
            tmp.program();
        }
        else
        {
            // Single-face tile: the full face row Y of every tile in the block.
            _llk_pack_untilize_program_row_mop_<MEGAROW>(MOP_OUTER_LOOP, MOP_INNER_LOOP, PACK_SEL(PACKCNT));
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
 * @tparam row_num_datums: Datums each tile contributes to an output row (TILE_C_DIM unless narrow_row); an
 *         output row of a block holds block_ct_dim * row_num_datums datums. With block_ct_dim != full_ct_dim
 *         the per-row L1 stride and the block start are kept in 16 B units, so block_ct_dim * row_num_datums
 *         datums must be a multiple of 16 bytes (asserted).
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note On the math thread, @ref _llk_math_eltwise_unary_datacopy_ (A2D) populates the dest register this packer reads.
 * @note Pair with @ref _llk_pack_untilize_uninit_ after the matching @ref _llk_pack_untilize_ execute calls.
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

    // The one-tile narrow MOP walks the rows itself, so ADDR_MOD_0 has to step Y after every PACR.
    constexpr bool step_y_per_pacr = _llk_pack_untilize_single_tile_v<block_ct_dim, full_ct_dim, narrow_row>;
    _llk_pack_untilize_configure_addrmod_<diagonal, step_y_per_pacr>();

    _llk_pack_untilize_mop_config_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(face_r_dim, num_faces);

    if (block_ct_dim != full_ct_dim)
    {
        // Match BH: 2 face-cols per tile-row for num_faces>=2; old WH `num_faces/2` gave 1 only at num_faces==2 (#52175).
        // A narrow row holds row_num_datums datums per tile.
        const std::uint32_t row_datums_per_tile = narrow_row ? row_num_datums : ((num_faces == 1) ? 1 : 2) * FACE_C_DIM;
        const std::uint32_t output_addr_offset  = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * row_datums_per_tile);
        // The per-row stride below and the caller's block start are kept in L1 address units: a narrow row of one
        // block that is not a multiple of the unit would be truncated and the rows would overlap. full_ct_dim is a
        // multiple of block_ct_dim, so this also covers the row stride.
        LLK_ASSERT(
            SCALE_DATUM_SIZE(pack_dst_format, block_ct_dim * row_datums_per_tile) % PACK_UNTILIZE_L1_ADDRESS_UNIT_BYTES == 0,
            "block_ct_dim * row_num_datums datums must be a multiple of 16 bytes for a partial block");
        TT_SETDMAREG(
            0,
            LOWER_HALFWORD(output_addr_offset / PACK_UNTILIZE_L1_ADDRESS_UNIT_BYTES),
            0,
            LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET)); // row offset in L1 address units
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
 * Programs the L1 destination address, then runs the MOP once per Dest row of the block
 * (@ref _llk_pack_untilize_num_rows_ runs), resetting the W counter before each run, advancing Y after it
 * and, for a partial block, replaying the per-row L1 address update; finally closes the block. A one-tile
 * narrow block (block_ct_dim == full_ct_dim == 1 with narrow_row) is packed by a single MOP run instead.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam diagonal: True to configure for diagonal (per-packer split) untilize packing.
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row; selects the one-tile fast path
 *         and must match the init.
 * @tparam row_num_datums: Datums each tile contributes to an output row (TILE_C_DIM unless narrow_row); must
 *         match the init.
 * @tparam tile_dst_ct_offset: Compile-time column-tile offset into the destination register.
 * @param address: L1 destination base address for the block.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param tile_dst_rt_offset: Runtime row-tile offset into the destination register.
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
    const std::uint32_t num_faces          = 4,
    const std::uint32_t tile_dst_rt_offset = 0)
{
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal, row_num_datums>(address, pack_dst_format);

    [[maybe_unused]] const std::uint32_t num_rows = _llk_pack_untilize_num_rows_(face_r_dim, num_faces);

    if constexpr (_llk_pack_untilize_single_tile_v<block_ct_dim, full_ct_dim, narrow_row>)
    {
        // The MOP programmed by init walks all num_rows rows of the one tile by itself; the close PACR below
        // clears Y for the next block.
        set_dst_write_addr(tile_dst_ct_offset + tile_dst_rt_offset); // Select the tile
        ckernel::ckernel_template::run();
    }
    else
    {
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
