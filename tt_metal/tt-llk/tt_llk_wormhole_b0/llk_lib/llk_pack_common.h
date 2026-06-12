// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "cpack_common.h"
#include "llk_defs.h"

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Stall the packer until the math thread has produced data to pack.
 *
 * Waits on the MATH_PACK semaphore so the packer does not run ahead of the math result.
 */
// wait until math is done and has produced something to pack
inline void _llk_packer_wait_for_math_done_()
{
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

/**
 * @brief Signal the MATH_PACK semaphore (decrement via SEMGET) to release the math thread.
 *
 * Tells math it may overwrite the destination register again now that the packer has consumed it.
 *
 * @tparam WaitRes: p_stall resource mask to stall on before signalling (e.g. p_stall::NONE, p_stall::PACK); default p_stall::NONE issues no stall.
 */
// Tell math that it can write again
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    t6_semaphore_get<WaitRes>(semaphore::MATH_PACK); // Indicate that packer is done and header is written into L1
}

/**
 * @brief Finish a destination-register section: wait for pack, clear dest, and release math.
 *
 * Stalls until the pack completes, zeroes the just-packed dest region (all of dest for SyncFull, the
 * active half for SyncHalf), then signals the MATH_PACK semaphore. For SyncHalf it also flips the
 * dest-offset id and re-selects the packer dest registers so the next half can be packed.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 */
// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); // wait for pack to finish

    if constexpr (Dst == DstSync::SyncFull)
    {
        constexpr std::uint32_t CLEAR_MODE = is_fp32_dest_acc_en ? p_zeroacc::CLR_ALL_32B : p_zeroacc::CLR_ALL;
        TTI_ZEROACC(CLEAR_MODE, ADDR_MOD_1, 0);
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        constexpr std::uint32_t CLEAR_MODE = is_fp32_dest_acc_en ? p_zeroacc::CLR_HALF_32B : p_zeroacc::CLR_HALF;
        TT_ZEROACC(CLEAR_MODE, ADDR_MOD_1, (dest_offset_id) % 2);
    }

    // Tell math that it can write again
    _llk_packer_set_math_semaphore_<p_stall::NONE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<Dst>();
    }
}

/**
 * @brief Initialize the packer destination-offset GPRs and select the dest registers.
 *
 * Programs the low/high dest-offset GPRs for row-major order and selects the packer destination
 * registers for the chosen sync mode.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @tparam diagonal: True to lay out dest offsets for diagonal (per-packer split) packing.
 * @param face_r_dim: Number of rows per face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 */
template <DstSync Dst, PackMode pack_mode = PackMode::Default, bool diagonal = false>
inline void _llk_init_packer_dest_offset_registers_(const std::uint32_t face_r_dim = FACE_R_DIM, const bool narrow_tile = false)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Wormhole B0 pack dest offset setup supports only PackMode::Default and PackMode::Untilize");
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish
    if constexpr (pack_mode == PackMode::Untilize)
    {
        const std::uint32_t face_r_offset = ((face_r_dim == 1) || narrow_tile || diagonal) ? FACE_R_DIM : (face_r_dim >> 1);
        if constexpr (diagonal)
        {
            // For example if face_offset = 8:
            //  Packer0 :  0,16,  1,17 ...  7, 23
            //  Packer1 :  8,24,  9,25 ... 15, 31
            //  Packer2 : 32,48, 33,49 ... 39, 55
            //  Packer3 : 40,56, 41,57 ... 47, 63
            TTI_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TT_SETDMAREG(0, 0x000 + 0x20 + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TT_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x20 + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
        }
        else
        {
            // For example if face_offset = 8:
            //  Packer0 :  0,16,  1,17 ...  7, 23
            //  Packer1 :  8,24,  9,25 ... 15, 31
            //  Packer2 : 32,48, 33,49 ... 39, 55
            //  Packer3 : 40,56, 41,57 ... 47, 63
            TTI_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TT_SETDMAREG(0, 0x000 + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(0, 0x000 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
            TT_SETDMAREG(0, 0x000 + 0x20 + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
            TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TT_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
            TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
            TT_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x20 + face_r_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
        }
    }
    else
    {
        // For non-untilize, faces are stored sparsely in dest register
        // Each face occupies FACE_R_DIM (16) rows regardless of actual face_r_dim
        // Face 0: rows 0-15 (data in 0 to face_r_dim-1, padding in face_r_dim to 15)
        // Face 1: rows 16-31
        // Face 2: rows 32-47
        // Face 3: rows 48-63
        // So face stride is always FACE_R_DIM, not face_r_dim
        TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
        TTI_SETDMAREG(0, 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1)); // FACE_R_DIM = 16 = 0x10
        TTI_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2)); // 2 * FACE_R_DIM
        TTI_SETDMAREG(0, 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3)); // 3 * FACE_R_DIM
        TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
        TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
        TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
        TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
    }
    select_packer_dest_registers<Dst>();
}

/**
 * @brief Initialize packer destination state at the start of a kernel.
 *
 * Syncs Tensix, resets the dest-offset id, programs the packer dest-offset registers, initializes the
 * packer address counter, and resets the tile dest pointer.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @param face_r_dim: Number of rows per face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 * @ref _llk_init_packer_dest_offset_registers_ performs the dest-offset register setup.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_dest_init_(const std::uint32_t face_r_dim = FACE_R_DIM, const bool narrow_tile = false)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Wormhole B0 pack dest init supports only PackMode::Default and PackMode::Untilize");
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst, pack_mode>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

/**
 * @brief Select which destination-register tile the packer reads from.
 *
 * Sets the packer CH0 W counter to tile_index, which addresses the tile within the destination
 * register that subsequent PACR instructions pack out.
 *
 * @param tile_index: Index of the source tile in the destination register.
 */
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
}

/**
 * @brief Configure the packer relu mode and threshold.
 *
 * Writes the relu mode and threshold carried by the config to the STACC_RELU config register.
 *
 * @param relu_config: Relu configuration carrying the mode and threshold value.
 */
TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    const std::uint32_t mode = static_cast<std::uint32_t>(relu_config.get_mode());
    const std::uint32_t val  = (relu_config.get_threshold() << STACC_RELU_ReluThreshold_SHAMT) | (mode << STACC_RELU_ApplyRelu_SHAMT);
    TT_SETDMAREG(0, val & 0xffff, 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, val >> 16, 0, HI_16(p_gpr_pack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

/**
 * @brief Enable or disable packer L1 accumulation.
 *
 * @param enable: Non-zero to accumulate packed output into existing L1 data, zero to overwrite.
 */
inline void _llk_pack_reconfig_l1_acc_(const std::uint32_t enable)
{
    reconfigure_packer_l1_acc(enable);
}

/**
 * @brief Configure the packer edge-offset masks and tile-row-set mapping for a reduce output.
 *
 * Programs PCK_EDGE_OFFSET_SEC0/SEC1 masks and TILE_ROW_SET_MAPPING_1 so that only the reduced
 * datums survive: for row reduce a single column per row, for col reduce only the first row, and for
 * scalar reduce a single datum, with per-packer selection appropriate to the reduce dimension.
 *
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @note Pairs with @ref _llk_math_reduce_ on the math thread, whose reduced output these masks gate.
 * @note Call @ref _llk_pack_reduce_mask_clear_ to restore the default pass-through masks.
 */
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_reduce_mask_config_(const std::uint32_t face_r_dim = FACE_R_DIM)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Wormhole B0 pack reduce-mask config supports only PackMode::Default and PackMode::Untilize");
    ckernel::packer::pck_edge_offset_u pack_edge_offset = {.val = 0};

    // We initialize PCK_EDGE_OFFSET_SEC0 mask to clear out all the datums in the row
    pack_edge_offset.f.mask             = 0x0;
    std::uint32_t row_set_mapping_1     = 0;
    std::uint32_t edge_offset_sec1_mask = 0;

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        // PCK_EDGE_OFFSET_SEC1 mask will clear out all the datums in the row except the first one

        // All packers use TILE_ROW_SET_MAPPING_1 to support both narrow tiles (packers 0,1)
        // and wide tiles (packers 0,2)
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;
        pack_edge_offset.f.tile_row_set_select_pack2 = 1;
        pack_edge_offset.f.tile_row_set_select_pack3 = 1;

        edge_offset_sec1_mask = 0x0001;
        if constexpr (pack_mode == PackMode::Untilize)
        {
            row_set_mapping_1 = 0x11111111; // each packer packs 1x32 row
        }
        else
        {
            // TILE_ROW_SET_MAPPING_1 configuration sets all rows to use PCK_EDGE_OFFSET_SEC1 mask
            row_set_mapping_1 = 0x55555555; // each packer packs 1x16 row
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        // PCK_EDGE_OFFSET_SEC1 mask will pass through all the datums in the row as they are
        edge_offset_sec1_mask = 0xffff;

        // Packer 0 and 1 will use TILE_ROW_SET_MAPPING_1, while packer 2 and 3 will keep using
        // TILE_ROW_SET_MAPPING_0 configuration which is the default one
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;

        if constexpr (pack_mode == PackMode::Untilize)
        {
            row_set_mapping_1 = 0x00000005; // each packer packs 1x32 row
        }
        else
        {
            // TILE_ROW_SET_MAPPING_1 configuration sets only first row to use PCK_EDGE_OFFSET_SEC1 mask
            row_set_mapping_1 = 0x00000001; // each packer packs 1x16 row
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        // PCK_EDGE_OFFSET_SEC1 mask will clear out all the datums in the row except the first one
        edge_offset_sec1_mask = 0x0001;
        // Packer 0  will use TILE_ROW_SET_MAPPING_1, while packers 1,2 and 3 will keep using
        // TILE_ROW_SET_MAPPING_0 configuration which is the default one
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;

        // TILE_ROW_SET_MAPPING_1 configuration sets only first row to use PCK_EDGE_OFFSET_SEC1 mask
        row_set_mapping_1 = 0x00000001;
    }

    // Initialize TMP registers with values we need to write in CFG registers
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, LOWER_HALFWORD(edge_offset_sec1_mask), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_SETDMAREG(0, LOWER_HALFWORD(row_set_mapping_1), 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, UPPER_HALFWORD(row_set_mapping_1), 0, HI_16(p_gpr_pack::TMP1));

    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(face_r_dim);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC1_pack_reads_per_xy_plane_RMW>(face_r_dim);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_RMW>(face_r_dim);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC3_pack_reads_per_xy_plane_RMW>(face_r_dim);

    // Configure packer
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    TTI_NOP;
    TTI_NOP;
}

/**
 * @brief Restore the default packer edge masks and tile-row-set mapping after a reduce.
 *
 * Resets the edge-offset masks to pass-through and points all tile-row-set mappings back to
 * PCK_EDGE_OFFSET_SEC0, undoing @ref _llk_pack_reduce_mask_config_.
 *
 * @note Pairs with @ref _llk_pack_reduce_mask_config_.
 */
inline void _llk_pack_reduce_mask_clear_()
{
    // By default, all packers are set to use TILE_ROW_SET_MAPPING_0 and
    // mask is configured to pass through all the datums
    pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask            = 0xffff;

    // Initialize TMP registers with values we need to write in CFG registers
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));

    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC1_pack_reads_per_xy_plane_RMW>(1);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_RMW>(1);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC3_pack_reads_per_xy_plane_RMW>(1);

    // Clear out packer configuration for reduce
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);

    // All mappings point to PCK_EDGE_OFFSET_SEC0_mask_ADDR32
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    TTI_NOP;
    TTI_NOP;
}
