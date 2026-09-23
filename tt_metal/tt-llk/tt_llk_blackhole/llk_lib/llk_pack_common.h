// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_mutex_guard.h"
#include "ckernel_ops.h"
#include "cpack_common.h"
#include "llk_defs.h"
#include "tensor_shape.h"

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
    // The mask must cover STALL_SYNC as well as STALL_TDMA: the Wait Gate only blocks classes named in it, so with a
    // TDMA-only mask the Sync-class ATGETM of a mutexed _llk_pack_ slips past this unmet wait and takes
    // mutex::THREAD2_ADC, the SETADC behind it blocks head-of-line, and the ATRELM is stranded behind that. The packer
    // then waits on MATH_PACK holding the mutex the unpack thread needs every iteration: deadlock. Naming the Sync
    // class costs nothing when unmutexed.
    TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
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
        TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
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
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK); // wait for pack to finish

    // RowMajor order
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
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
 * @ref _llk_init_packer_dest_offset_registers_ performs the dest-offset register setup.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>();
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

/**
 * @brief Select which destination-register tile the packer reads from.
 *
 * Sets the packer CH0 W counter to tile_index, which addresses the tile within the destination
 * register that subsequent PACR instructions pack out.
 *
 * @tparam mutex_ADC: When true, serialize the SETADC issue against mutex::THREAD2_ADC. Needed only when
 *         another thread borrows the pack thread's ADCs; forwarded from @ref _llk_pack_.
 * @param tile_index: Index of the source tile in the destination register.
 */
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    T6MutexLockGuard<mutex_ADC> guard(mutex::THREAD2_ADC);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
}

/**
 * @brief Configure the packer relu mode and threshold.
 *
 * Writes the relu mode and threshold carried by the config to the STACC_RELU config register.
 *
 * @param relu_config: Relu configuration supplying the mode and threshold to apply.
 */
TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    const std::uint32_t mode = static_cast<std::uint32_t>(relu_config.get_mode());
    const std::uint32_t val  = (relu_config.get_threshold() << STACC_RELU_ReluThreshold_SHAMT) | (mode << STACC_RELU_ApplyRelu_SHAMT);

    // STACC_RELU shares this cfg word with ALU_ACC_CTRL_Zero_Flag_disabled_src/dst (bits 0-1, owned by the
    // MATH/UNPACK threads). A whole-word WRCFG_32b would clobber those bits, so use a masked RMW under
    // mutex::REG_RMW -- matching how configure_pack programs relu.
    static_assert(STACC_RELU_ApplyRelu_ADDR32 == STACC_RELU_ReluThreshold_ADDR32, "STACC_RELU ApplyRelu and ReluThreshold must share ADDR32 for combined RMW");
    constexpr std::uint32_t hw_relu_mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;

    // Only the packer needs draining: RMWCIB takes the data as an immediate (no GPR), so there is no
    // SETDMAREG->WRCFG producer to fence -- the THCON wait the old whole-word path used is no longer needed.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    t6_mutex_acquire(mutex::REG_RMW);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(val);
    t6_mutex_release(mutex::REG_RMW);
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
 * @brief Configure the packer face, row, and edge masks for a reduce output.
 *
 * Programs PCK_EDGE_OFFSET_SEC0/SEC1 masks and TILE_ROW_SET_MAPPING_1 so that only the reduced
 * datums survive: for row reduce a single column per row, for col reduce only the first row, and for
 * scalar reduce a single datum. Default tiled packing selects the row mask per face through
 * TILE_FACE_SET_MAPPING_0; Blackhole has one packer, so the packer selectors cannot select faces.
 *
 * @tparam reduce_type: Pool type; MAX selects negative-infinity mode, except BFP outputs retain zero fill.
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @param pack_dst_format: Packer output (L1) data format, as last programmed by the caller's pack reconfig.
 * @param tensor_shape: Output face dimensions and face grid.
 * @note Untilize retains its existing row-mask configuration; face selection below is for Default only.
 * @note Pairs with @ref _llk_math_reduce_ on the math thread, whose reduced output these masks gate.
 * @note Call @ref _llk_pack_reduce_mask_clear_ to restore the default pass-through masks.
 */
template <PoolType reduce_type, ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_reduce_mask_config_(const std::uint32_t pack_dst_format, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    ckernel::packer::pck_edge_offset_u pack_edge_offset = {.val = 0};

    // PCK_EDGE_OFFSET_SEC0 masks every datum in the row with the selected fill value.
    pack_edge_offset.f.mask             = 0x0;
    std::uint32_t row_set_mapping_1     = 0;
    std::uint32_t edge_offset_sec1_mask = 0;

    // The direct selector is used when face lookup is disabled (Untilize).
    pack_edge_offset.f.tile_row_set_select_pack0 = 1;

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        // PCK_EDGE_OFFSET_SEC1 masks every datum in the row except the first one
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
        // PCK_EDGE_OFFSET_SEC1 masks every datum in the row except the first one
        edge_offset_sec1_mask = 0x0001;
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

    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(tensor_shape.face_r_dim);

    // Configure packer
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    if constexpr (reduce_type == PoolType::MAX)
    {
        // Masked infinities can overwrite the shared BFP exponent and zero the valid result.
        cfg_reg_rmw_tensix<PCK_EDGE_MODE_mode_RMW>(!IS_BFP_FORMAT(pack_dst_format));
    }

    if constexpr (pack_mode == PackMode::Default)
    {
        // The constants repeat 2-bit row-table selectors across all 16 face-table entries:
        // 0x55555555 = [1], 0x11111111 = [1,0], 0x05050505 = [1,1,0,0], 0x01010101 = [1,0,0,0].
        // Row-table 1 applies the reduction mask; row-table 0 masks the entire face.
        const std::uint32_t face_set_mapping = [&tensor_shape]
        {
            static_assert(dim == ReduceDim::REDUCE_ROW || dim == ReduceDim::REDUCE_COL || dim == ReduceDim::REDUCE_SCALAR, "Invalid reduction dimension");
            if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 1)
            {
                return 0x55555555;
            }
            else if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2)
            {
                return dim == ReduceDim::REDUCE_COL ? 0x55555555 : 0x11111111;
            }
            else if (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 1)
            {
                return dim == ReduceDim::REDUCE_ROW ? 0x55555555 : 0x11111111;
            }
            else
            {
                LLK_ASSERT(tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 2, "Invalid tensor shape face grid");
                if constexpr (dim == ReduceDim::REDUCE_ROW)
                {
                    return 0x11111111;
                }
                else if constexpr (dim == ReduceDim::REDUCE_COL)
                {
                    return 0x05050505;
                }
                else
                {
                    return 0x01010101;
                }
            }
        }();
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
        cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, 0, 0xffffffff>(face_set_mapping);
        // Select face table 0 for packer 0 and enable the face -> row -> column-mask lookup.
        cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100);
    }
    else
    {
        cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_enable_RMW>(0);
    }

    TTI_NOP;
    TTI_NOP;
}

/**
 * @brief Restore the default packer edge masks and tile-row-set mapping after a reduce.
 *
 * Disables face selection, resets the edge-offset masks to pass-through, and points the active
 * tile-row-set mappings back to PCK_EDGE_OFFSET_SEC0, undoing @ref _llk_pack_reduce_mask_config_.
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
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_enable_RMW>(0);

    // Clear out packer configuration for reduce
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);

    // All mappings point to PCK_EDGE_OFFSET_SEC0_mask_ADDR32
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    TTI_NOP;
    TTI_NOP;
}
