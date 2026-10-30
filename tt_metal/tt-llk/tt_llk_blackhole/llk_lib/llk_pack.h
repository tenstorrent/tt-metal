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

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 *
 * Programs ADDR_MOD_0/1/2 with the src Y and Z increment/clear patterns the pack MOP relies
 * on to traverse the destination register and step through faces for the given layout.
 * The dest (channel-1) counters are left alone: output placement comes from L1_Dest_addr and
 * the packer's own write pointer, and channel-1 strides are zero outside fast-untilize.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    if constexpr (pack_mode == PackMode::Untilize)
    {
        /*  Y src inc by 1 to give strided increments:
            Rows: 0, 16, 1, 17, 2, 18, ........ 15, 31
        */
        addr_mod_pack_t {.y_src = {.incr = 1}, .z_src = {.incr = 0}}.set(ADDR_MOD_0);

        /* Increment Faces by 2 to give next 2 faces:
            Rows: 32, 48, 33, 49, 34, 50........47, 63
        */
        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_1);

        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}}.set(ADDR_MOD_2);
    }
    else if constexpr (pack_mode == PackMode::Tilize)
    {
        addr_mod_pack_t {.y_src = {.incr = 4}, .z_src = {.incr = 0}}.set(ADDR_MOD_0);

        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 0}}.set(ADDR_MOD_1);

        // Increment faces by 2 (jump 2 dest address 32)
        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 4},
        }
            .set(ADDR_MOD_0);

        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_1);

        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 1, .clr = 0},
        }
            .set(ADDR_MOD_2);
    }
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 *
 * Programs the ckernel MOP (and, for tilize, a replay buffer) with the PACR instruction sequence
 * that packs one tile worth of data, selecting packer interfaces and ADDR_MODs per the layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note @ref _llk_pack_configure_addrmod_ must have programmed the ADDR_MOD slots for the same pack_mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    constexpr std::uint32_t MEGAROW          = 1;
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    if constexpr (pack_mode == PackMode::Untilize)
    {
        const std::uint32_t PACK_INTF_SEL  = (tile_c_dim < TILE_C_DIM) ? p_pacr::SINGLE_INTF_ACTIVE : p_pacr::TWO_INTFS_ACTIVE;
        const std::uint32_t MOP_INNER_LOOP = face_r_dim;
        const std::uint32_t MOP_OUTER_LOOP = (tile_c_dim < TILE_C_DIM) ? num_faces : (num_faces >> 1);

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL,
                0,
                MEGAROW,
                p_pacr::NO_CTXT_CTRL,
                0,
                0));

        tmp.set_last_inner_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            MEGAROW,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1));
        tmp.program();
    }
    else if constexpr (pack_mode == PackMode::Tilize)
    {
        const std::uint32_t PACK_INTF_SEL_0 = 0b0101;
        const std::uint32_t PACK_INTF_SEL_1 = 0b1010;
        const std::uint32_t MOP_INNER_LOOP  = 1;
        const std::uint32_t MOP_OUTER_LOOP  = (num_faces > 1) ? (num_faces >> 1) : 1;

        if (face_r_dim == 1)
        {
            // Single-row faces (e.g. a 1x32 tile = two 1x16 faces). The tileize unpacker splits the
            // single source row into two consecutive DEST rows (row 0 -> face 0, row 1 -> face 1). With
            // one row per face the two faces are contiguous in L1 ([face 0 | face 1] == one 1x32 row), so
            // a single PACR with two active read interfaces (interface k reads DEST row base + k) packs
            // both faces at once - the row-interleave the replay buffer un-swizzles for face_r_dim >= 2
            // collapses to these two rows. The replay-buffer path cannot express this anyway
            // (replay_buf_len = face_r_dim - 1 = 0 is an invalid replay length, and num_instrs_per_face
            // underflows).
            LLK_ASSERT(num_faces == 2, "Tilize with face_r_dim == 1 supports num_faces == 2 only");
            ckernel::ckernel_template tmp(
                MOP_OUTER_LOOP,
                MOP_INNER_LOOP,
                // Interfaces 0 and 1 read DEST rows 0 and 1 (face 0 and face 1); ADDR_MOD_2, close the tile.
                TT_OP_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_2,
                    p_pacr::ADDR_CNT_CTXT_0,
                    ZERO_OUTPUT_FLAG,
                    p_pacr::TWO_INTFS_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    1));
            tmp.program();
            return;
        }

        // Last row of half-tile (face_r_dim rows) is different between halves, so can't be replayed.
        LLK_ASSERT(face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be 2, 4, 8, or 16 for tilize");
        const std::uint32_t replay_buf_len = face_r_dim - 1;

        // This replay buffer finishes 2 faces
        load_replay_buf(
            0,
            replay_buf_len,
            // Lambda function to set up replay buffer
            [PACK_INTF_SEL_0, PACK_INTF_SEL_1, ZERO_OUTPUT_FLAG, MEGAROW, face_r_dim]
            {
                // Number of instructions per face (minus 1 for the last special instruction)
                const std::uint32_t num_instrs_per_face = (face_r_dim >> 1) - 1;

                // Face 0 -> mask rows 1010
                // First num_instrs_per_face instructions use ADDR_MOD_0
                for (std::uint32_t i = 0; i < num_instrs_per_face; i++)
                {
                    TTI_PACR(
                        p_pacr::CFG_CTXT_0,
                        p_pacr::NO_ROW_PAD_ZERO,
                        p_pacr::DST_ACCESS_NORMAL_MODE,
                        ADDR_MOD_0,
                        p_pacr::ADDR_CNT_CTXT_0,
                        ZERO_OUTPUT_FLAG,
                        PACK_INTF_SEL_0,
                        0,
                        MEGAROW,
                        p_pacr::NO_CTXT_CTRL,
                        0,
                        0);
                }
                // Last instruction for Face 0 uses ADDR_MOD_1
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_1,
                    p_pacr::ADDR_CNT_CTXT_0,
                    ZERO_OUTPUT_FLAG,
                    PACK_INTF_SEL_0,
                    0,
                    MEGAROW,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);

                // Face 1 -> mask rows 0101
                // num_instrs_per_face instructions use ADDR_MOD_0
                // The last instruction is handled separately outside the replay buffer
                for (std::uint32_t i = 0; i < num_instrs_per_face; i++)
                {
                    TTI_PACR(
                        p_pacr::CFG_CTXT_0,
                        p_pacr::NO_ROW_PAD_ZERO,
                        p_pacr::DST_ACCESS_NORMAL_MODE,
                        ADDR_MOD_0,
                        p_pacr::ADDR_CNT_CTXT_0,
                        ZERO_OUTPUT_FLAG,
                        PACK_INTF_SEL_1,
                        0,
                        MEGAROW,
                        p_pacr::NO_CTXT_CTRL,
                        0,
                        0);
                }
                // Last PACR instruction of the half-tile must go separately in the MOP. This is to be able to override it, to ensure that for the second half
                // the tile is closed correctly.
            });

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            lltt::replay_insn(0, replay_buf_len),
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_2,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL_1,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                0) // don't close tile
        );

        // Close the tile only when it is actually done.
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL_1,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1));

        tmp.program();
    }
    else
    {
        const std::uint32_t PACK_INTF_SEL =
            face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

        const std::uint32_t MOP_INNER_LOOP = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
        const std::uint32_t MOP_OUTER_LOOP = num_faces * num_tiles;

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL,
                0,
                0,
                0,
                0,
                0));
        tmp.set_last_inner_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            1));

        // if (partial_face) {
        //     tmp.set_start_op(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0,
        //     ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Don't close the tile, point to the next face
        //     tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0)); // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
        //     tmp.set_loop_op1(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0,
        //     ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Close the tile
        // }

        tmp.program();
    }
}

namespace llk_pack_internal_bh
{
/**
 * @brief Shared packer-init body behind the @ref _llk_pack_init_ overloads.
 *
 * Runs the common init sequence with each stage individually skippable: program the ADDR_MOD slots,
 * build the MOP template, set the packer strides, and program the packer X (datum) counter. The
 * skip_* template flags let a caller reuse state already established by a prior init or hw-configure.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, the packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave the ADDR_MOD slots untouched.
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @tparam mutex_ADC: When true, serialize the SETADCXX issue against mutex::THREAD2_ADC. Needed only
 *         when another thread borrows the pack thread's ADCs.
 * @param pack_src_format: Source (dest register) data format; only used when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note Init owns the packer X (datum) counter (SETADCXX): every init programs its own value, mirroring
 *       the Wormhole contract. On Blackhole the value is always a single row (FACE_C_DIM - 1).
 * @note mutex_ADC has to match the value given to @ref _llk_pack_ ; guarding only one of the two
 *       leaves unguarded SETADC issues. Supported for PackMode::Default only.
 */
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    static_assert(
        !mutex_ADC || pack_mode == PackMode::Default,
        "mutex_ADC is supported for PackMode::Default only: other modes issue ADC instructions from inside the MOP, "
        "which the mutex cannot cover");

    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);
    }

    // Program the packer X (datum) counter. Per the "inits own SETADCXX" contract, every init sets its
    // own value; on Blackhole x_start/x_end must stay within a single row (0..FACE_C_DIM-1).
    {
        T6MutexLockGuard<mutex_ADC> guard(mutex::THREAD2_ADC);
        TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    }
}
} // namespace llk_pack_internal_bh

/**
 * @brief Reconfigure the packer source/destination data formats and tile geometry at runtime.
 *
 * Used to switch the packer to a new data format without a full HW re-configure.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
}

/**
 * @brief One-time hardware configuration of the packer for a given data format and tile geometry.
 *
 * Programs the packer config registers (formats, strides, relu) for the chosen pack mode. Call once
 * before the init/execute sequence, or again when a later op needs a different packer configuration.
 *
 * @note Drains the packer and blocks this RISC until it is idle before any register is rewritten, so it
 *       is safe to call while a pack from the previous op is still in flight.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param relu_config: Packed relu mode and threshold configuration (0 disables relu).
 * @note For 8-bit unpack-source datums, do not use PackMode::Tilize: the Blackhole row-unswizzling workaround is skipped (and is unnecessary, as 8-bit formats
 * are unaffected by the issue).
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t tile_c_dim  = TILE_C_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const std::uint32_t relu_config = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    LLK_ASSERT_BLOCK(llk_assert_fp32_dest_acc_en = is_fp32_dest_acc_en);
    configure_pack<is_fp32_dest_acc_en, pack_mode>(pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, relu_config);
}

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op given a source format.
 *
 * The single Blackhole pack-init entry point: programs ADDR_MODs, the MOP template, (unless skipped)
 * the packer strides, and the packer X (datum) counter. Per the "inits own SETADCXX" contract, every
 * init sets its own X-counter value (FACE_C_DIM - 1 on Blackhole). When packing tilized 8-bit datums
 * the Blackhole row-unswizzle workaround can be skipped (the issue does not affect 8-bit datums).
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: When true, do not re-program the packer strides (e.g. when a prior
 *         hw-configure / reconfig already established them, or the caller programs them itself).
 * @tparam mutex_ADC: When true, serialize the SETADCXX issue against mutex::THREAD2_ADC. Needed only
 *         when another thread borrows the pack thread's ADCs. Exposed upstream as
 *         llk_pack_init_mutex_ADC / pack_init_mutex_ADC.
 * @param pack_src_format: Source (dest register) data format. Only consulted when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @param skip_bh_tilize_workaround: When true (8-bit src datums), skip the Blackhole tilize row-unswizzle workaround.
 * @note Pair with @ref _llk_pack_uninit_ after the matching @ref _llk_pack_ execute calls.
 * @note mutex_ADC has to match the value given to @ref _llk_pack_ ; guarding only one of the two
 *       leaves unguarded SETADC issues. Supported for PackMode::Default only: the other modes embed ADC
 *       instructions in the MOP itself (e.g. the Tilize MOP's SETADCZW end op), which execute inside
 *       ckernel_template::run() where a scope guard cannot reach them.
 */
template <
    PackMode pack_mode       = PackMode::Default,
    bool zero_output         = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false,
    bool mutex_ADC           = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool skip_bh_tilize_workaround)
{
    static_assert(
        !mutex_ADC || pack_mode == PackMode::Default,
        "mutex_ADC is supported for PackMode::Default only: other modes issue ADC instructions from inside the MOP, "
        "which the mutex cannot cover");

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const DataFormat src_format = static_cast<DataFormat>(pack_src_format);
    if (src_format == DataFormat::Float32)
    {
        LLK_ASSERT(num_tiles <= 4, "Max supported num_tiles for FLOAT32 is 4.");
    }
    else if ((src_format == DataFormat::Float16) || (src_format == DataFormat::Float16_b))
    {
        LLK_ASSERT(num_tiles <= 8, "Max supported num_tiles for FLOAT16 or FLOAT16_B is 8.");
    }

    // 8bit datums in the unpack src format are not affected by the blackhole issue,
    // so we can skip the workaround which involves unswizzling rows in the tile.
    if (skip_bh_tilize_workaround && pack_mode == PackMode::Tilize)
    {
        llk_pack_internal_bh::pack_init_apply<PackMode::Default, zero_output, skip_addrmod_config, skip_packer_strides, mutex_ADC>(
            pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
    }
    else
    {
        llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, mutex_ADC>(
            pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
    }
}

/**
 * @brief Tear down the packer after a pack op (no-op on Blackhole).
 *
 * On Blackhole @ref _llk_pack_init_ always sets the PAC X counter to FACE_C_DIM - 1 (a single row),
 * which is also its default value, so there is no per-pack state for this teardown to restore.
 *
 * @note Pairs with @ref _llk_pack_init_.
 */
inline void _llk_pack_uninit_()
{
    // No state to restore - Blackhole pack_init sets PAC X counter to FACE_C_DIM - 1 which is the default.
}

/**
 * @brief Pack one tile from the destination register to an L1 address.
 *
 * Selects the source dest tile, programs the L1 destination address, runs the packer MOP, and resets
 * the Z counters afterward.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (Tilize not supported here)
 * @tparam mutex_ADC: When true, serialize the per-tile SETADC issues against mutex::THREAD2_ADC. Needed
 *         only when another thread borrows the pack thread's ADCs. Exposed upstream as
 *         llk_pack_mutex_ADC / pack_tile_mutex_ADC.
 * @param tile_index: Index of the source tile in the destination register.
 * @param address: L1 destination address for the packed tile.
 * @note Call @ref _llk_pack_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_uninit_ once all pack calls are complete.
 * @note mutex_ADC has to match the value given to @ref _llk_pack_init_ ; guarding only one of the two
 *       leaves unguarded SETADC issues. Supported for PackMode::Default only, see @ref _llk_pack_init_.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false, bool rmw_dest = (Dst == DstSync::SyncFull && pack_mode == PackMode::Default)>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Blackhole: _llk_pack_ supports PackMode::Default and PackMode::Untilize only");
    static_assert(
        !mutex_ADC || pack_mode == PackMode::Default,
        "mutex_ADC is supported for PackMode::Default only: other modes issue ADC instructions from inside the MOP, "
        "which the mutex cannot cover");

    set_dst_write_addr<mutex_ADC>(tile_index);

    program_packer_destination<!rmw_dest>(address);

    ckernel::ckernel_template::run();

    {
        T6MutexLockGuard<mutex_ADC> guard(mutex::THREAD2_ADC);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
    }
}

/**
 * Pack num_tiles consecutive 32x32 tiles to consecutive L1 tiles with one run of the PackMode::Default program.
 * @tparam pack_mode: Packing layout, must be PackMode::Default.
 * @param num_tiles: Number of consecutive tiles to pack; start_tile_index + num_tiles must fit the dest sync region.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool rmw_dest = false>
inline void _llk_pack_block_(const std::uint32_t start_tile_index, const std::uint32_t address, const std::uint32_t num_tiles)
{
    LLK_ASSERT_PACK_NOT_FAST_UNTILIZE();
    static_assert(pack_mode == PackMode::Default, "Blackhole: _llk_pack_block_ supports PackMode::Default only");

    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    set_dst_write_addr(start_tile_index);

    program_packer_destination<!rmw_dest>(address);

    // Template 1 MOP: non-zero bits 19:10 override the programmed outer loop count for this run only.
    const std::uint32_t outer_loop_len = TILE_NUM_FACES * num_tiles;
    TT_MOP(1, outer_loop_len >> 6, (outer_loop_len & 0x3F) << 10);

    // The opt-in destination write leaves the Z counters to the last PACR's ADDR_MOD (see _llk_pack_rmw_dest_init_)
    if constexpr (!rmw_dest)
    {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
}

/**
 * Pack num_tiles consecutive tiles to L1 tiles tile_stride bytes apart, closing every tile so block-float and padded pages stay intact.
 * @param num_tiles: Number of consecutive tiles to pack; start_tile_index + num_tiles must fit the dest sync region.
 * @param tile_stride: Bytes from one L1 tile to the next, a multiple of 16 below 64 KB.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool rmw_dest = false>
inline void _llk_pack_block_closed_(
    const std::uint32_t start_tile_index,
    const std::uint32_t address,
    const std::uint32_t num_tiles,
    const std::uint32_t tile_stride,
    const std::uint32_t num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT_PACK_NOT_FAST_UNTILIZE();
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");
    LLK_ASSERT((tile_stride & 0xF) == 0 && tile_stride < 0x10000, "tile_stride must be a multiple of 16 below 64 KB");

    set_dst_write_addr(start_tile_index);

    program_packer_destination<!rmw_dest>(address);

    // The output address adds the packer W counter of channel 1 times this stride; no other pack program moves it.
    // Rewriting an unchanged stride cost the call up to 8 cycles.
    if (tile_stride != pack_ch1_w_stride)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
        pack_ch1_w_stride = tile_stride;
    }

    for (std::uint32_t tile = 0; tile < num_tiles; tile++)
    {
        TT_MOP(1, 0, num_faces << 10);
        TTI_INCADCZW(p_setadc::PAC, 1, 0, 1, 0);
    }

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1101);
}

namespace pack_block_dense_internal
{
// Replay slots 17 and up: the closed fast block records slots 0 to 16 and small tiles never use it.
constexpr std::uint32_t REPLAY_START = 17;

inline std::uint32_t pacr_word(const std::uint32_t face_r_dim, const std::uint32_t addr_mod, const std::uint32_t last)
{
    const std::uint32_t intf = face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        intf,
        0,
        0,
        0,
        0,
        last);
}

inline std::uint32_t pacrs_per_tile(const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    return num_faces * (face_r_dim < 4 ? 1 : face_r_dim >> 2);
}
} // namespace pack_block_dense_internal

/**
 * Opt-in setup of _llk_pack_block_dense_ for tiles of one to four faces of 1 to 16 rows, at most eight PACRs (a 16x32 tile), known at run time.
 * Call it after the default pack init, and again after any program that replaces the replay buffer.
 * Records one tile: its PACRs (the last one steps to the next face slot without closing), the step to the next tile in DEST and the face reset.
 */
inline void _llk_pack_block_dense_init_(const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    using namespace pack_block_dense_internal;
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t pacrs = pacrs_per_tile(face_r_dim, num_faces);
    LLK_ASSERT(pacrs >= 1 && pacrs <= 8, "The dense block pack records at most eight PACRs per tile");
    const std::uint32_t per_face = pacrs / num_faces;
    const std::uint32_t row      = pacr_word(face_r_dim, ADDR_MOD_0, 0);
    const std::uint32_t face     = pacr_word(face_r_dim, ADDR_MOD_2, 0);

    lltt::record<lltt::NoExec>(REPLAY_START, pacrs + 2);
    for (std::uint32_t f = 0; f < num_faces; f++)
    {
        for (std::uint32_t r = 1; r < per_face; r++)
        {
            TT_INSN(row);
        }
        TT_INSN(face);
    }
    TTI_INCADCZW(p_setadc::PAC, 0, 0, 1, 0);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
}

/**
 * Pack num_tiles consecutive small tiles to dense L1, each tile right after the previous one, with one close (Last) for the whole run
 * (needs _llk_pack_block_dense_init_ for the same tile shape). The per-tile tile close and the data converter restart of the closed block
 * are paid once. Not for block-float outputs and padded pages: use _llk_pack_block_closed_ there.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_dense_(
    const std::uint32_t start_tile_index,
    const std::uint32_t address,
    const std::uint32_t num_tiles,
    const std::uint32_t face_r_dim,
    const std::uint32_t num_faces)
{
    using namespace pack_block_dense_internal;
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    const std::uint32_t pacrs = pacrs_per_tile(face_r_dim, num_faces);

    set_dst_write_addr(start_tile_index);

    program_packer_destination(address);

    const std::uint32_t tile_replay = lltt::replay_insn(REPLAY_START, pacrs + 2);
    for (std::uint32_t tile = 1; tile < num_tiles; tile++)
    {
        TT_INSN(tile_replay);
    }
    // The last tile closes on its last PACR
    if (pacrs > 1)
    {
        TT_INSN(lltt::replay_insn(REPLAY_START, pacrs - 1));
    }
    TT_INSN(pacr_word(face_r_dim, ADDR_MOD_1, 1));
    TTI_INCADCZW(p_setadc::PAC, 0, 0, 1, 0);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

/**
 * Opt-in setup of the byte-wise destination write of _llk_pack_, _llk_pack_block_ and _llk_pack_block_closed_ (the rmw_dest template argument):
 * three RMWCIB over the low bytes of L1_Dest_addr instead of SETDMAREG, STALLWAIT, WRCFG, SETDMAREG and DMANOP. Call it after _llk_pack_init_ and
 * again after any program that replaces the pack MOP: it sets bit 31 of the register once and clears the Z counters, which the default block pack
 * then no longer resets after each call. The write is faster per tile where the packer is the bound and slower where the pack thread's config
 * writes contend with the unpacker's (latency-bound matmul and eltwise kernels), so it is not the default.
 */
inline void _llk_pack_rmw_dest_init_()
{
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TTI_RMWCIB3(0xff, 0x80, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
}

template <bool zero_output, std::uint32_t addr_mod, std::uint32_t last>
TT_ALWAYS_INLINE void pack_block_fast_pacr_()
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        ZERO_OUTPUT_FLAG,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        last);
}

/**
 * Opt-in setup of the fast block packs for kernels whose pack output is four faces of 16 rows, known at compile time.
 * Call it after the default pack init, and again after any program that replaces the pack MOP or the replay buffer.
 * Sets bit 31 of the destination register once and records one closed tile (16 PACRs, the last one closing the tile, then the W step).
 */
template <bool zero_output = false>
inline void _llk_pack_block_fast_init_()
{
    TTI_RMWCIB3(0xff, 0x80, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    load_replay_buf(
        0,
        17,
        []
        {
            for (int face = 0; face < 3; face++)
            {
                pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
                pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
                pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
                pack_block_fast_pacr_<zero_output, ADDR_MOD_2, 0>();
            }
            pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
            pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
            pack_block_fast_pacr_<zero_output, ADDR_MOD_0, 0>();
            pack_block_fast_pacr_<zero_output, ADDR_MOD_1, 1>();
            TTI_INCADCZW(p_setadc::PAC, 1, 0, 1, 0);
        });
}

/**
 * Same as _llk_pack_block_ with the destination written by bytes (needs _llk_pack_block_fast_init_).
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_fast_(const std::uint32_t start_tile_index, const std::uint32_t address, const std::uint32_t num_tiles)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    set_dst_write_addr(start_tile_index);

    program_packer_destination<false>(address);

    // Template 1 MOP: non-zero bits 19:10 override the programmed outer loop count for this run only.
    const std::uint32_t outer_loop_len = TILE_NUM_FACES * num_tiles;
    TT_MOP(1, outer_loop_len >> 6, (outer_loop_len & 0x3F) << 10);
    // No Z reset: the last PACR's ADDR_MOD clears the source Z and nothing in this program moves the L1 side counters.
}

/**
 * Same as _llk_pack_block_closed_ for four-face tiles of 16 rows, each tile replayed from the buffer (needs _llk_pack_block_fast_init_).
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_closed_fast_(
    const std::uint32_t start_tile_index, const std::uint32_t address, const std::uint32_t num_tiles, const std::uint32_t tile_stride)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");
    LLK_ASSERT((tile_stride & 0xF) == 0 && tile_stride < 0x10000, "tile_stride must be a multiple of 16 below 64 KB");

    set_dst_write_addr(start_tile_index);

    program_packer_destination<false>(address);

    if (tile_stride != pack_ch1_w_stride)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
        pack_ch1_w_stride = tile_stride;
    }

    for (std::uint32_t tile = 0; tile < num_tiles; tile++)
    {
        lltt::replay(0, 17);
    }

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1101);
}

/**
 * Opt-in W offset block packs: the output address is programmed once and each call moves it with the packer W counter of channel 1 (SETADC), so a call
 * writes no L1_Dest_addr. A config write waits for the packer behind it to drain, which costs a call of a few tiles 10 to 30 cycles.
 * Call after _llk_pack_block_fast_init_ (the closed form replays its tile).
 * @param address: L1 address (16 byte units) of the first output tile of the buffer.
 * @param tile_stride: Bytes from one output tile to the next, a multiple of 16 below 64 KB.
 */
inline void _llk_pack_block_w_init_(const std::uint32_t address, const std::uint32_t tile_stride)
{
    LLK_ASSERT((tile_stride & 0xF) == 0 && tile_stride < 0x10000, "tile_stride must be a multiple of 16 below 64 KB");
    program_packer_destination<false>(address);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
    pack_ch1_w_stride = tile_stride;
}

/**
 * Same as _llk_pack_block_fast_ with the output at out_tile tiles from the address of _llk_pack_block_w_init_.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_w_(const std::uint32_t start_tile_index, const std::uint32_t out_tile, const std::uint32_t num_tiles)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    set_dst_write_addr(start_tile_index);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_1, p_setadc::SET_W, out_tile);

    // Template 1 MOP: non-zero bits 19:10 override the programmed outer loop count for this run only.
    const std::uint32_t outer_loop_len = TILE_NUM_FACES * num_tiles;
    TT_MOP(1, outer_loop_len >> 6, (outer_loop_len & 0x3F) << 10);
    // Back to W 0 so that packs without the W offset write where they program
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1101);
}

/**
 * Same as _llk_pack_block_closed_fast_ with the output at out_tile tiles from the address of _llk_pack_block_w_init_ (its tile stride is the closed one).
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_closed_w_(const std::uint32_t start_tile_index, const std::uint32_t out_tile, const std::uint32_t num_tiles)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    set_dst_write_addr(start_tile_index);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_1, p_setadc::SET_W, out_tile);

    for (std::uint32_t tile = 0; tile < num_tiles; tile++)
    {
        lltt::replay(0, 17);
    }

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1101);
}

/**
 * Same as _llk_pack_block_closed_ (one MOP run per tile, any tile shape) with the output at out_tile tiles from the address of _llk_pack_block_w_init_.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_closed_w_mop_(
    const std::uint32_t start_tile_index, const std::uint32_t out_tile, const std::uint32_t num_tiles, const std::uint32_t num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be at least 1");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        ((start_tile_index + num_tiles) <= get_dest_max_tiles<Dst, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "The block exceeds the destination register sync region");

    set_dst_write_addr(start_tile_index);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_1, p_setadc::SET_W, out_tile);

    for (std::uint32_t tile = 0; tile < num_tiles; tile++)
    {
        TT_MOP(1, 0, num_faces << 10);
        TTI_INCADCZW(p_setadc::PAC, 1, 0, 1, 0);
    }

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1101);
}

