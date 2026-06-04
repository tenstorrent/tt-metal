// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Unpack.
//
// This is the standard unpack_A face stream specialized for fast_untilize. It
// still emits the same real SrcA face payload as regular unpack_A, but the
// common 16-bit DEST path skips the generic unpack_A zero-SrcB dvalid sideband.
// Native fp32 DEST is the exception: math uses ELWADD as SrcA + zero-SrcB, so
// unpack emits one valid zero SrcB face alongside each real SrcA face. Compressed
// BFP inputs use a unit_dim=1 MOP internally because each tile must be addressed
// explicitly to skip exponent sections, while the block-level chunk size still
// follows the 2/3/4-tile fast-untilize decomposition.

#pragma once

#include <cstdint>

#include "llk_unpack_A.h"

namespace ckernel
{

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_mop_config_(const std::uint32_t unit_dim)
{
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    static constexpr std::uint32_t unpack_srca            = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    const std::uint32_t outerloop     = unit_dim * 4;
    constexpr std::uint32_t innerloop = 1;

    // One MOP covers every face in the chunk: unit_dim tiles x 4 faces/tile.
    // The 16-bit DEST path copies SrcA to DEST with MOVA2D, so the loop body
    // only needs real SrcA UNPACRs. The fp32 DEST path copies with ELWADD,
    // which is a binary op and therefore needs a dummy zero SrcB dvalid for
    // every real SrcA face.
    if constexpr (is_fp32_dest_acc_en)
    {
        // Native fp32 DEST math uses ELWADD as a SrcA->DEST copy, which needs
        // a valid zero SrcB face alongside each real SrcA face.
        ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srcb_set_dvalid);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.program();
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_init_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim)
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(0, 0, FACE_R_DIM, 4, unpack_src_format, unpack_dst_format);
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(init_unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_reinit_unit_dim_(const std::uint32_t unit_dim)
{
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(unit_dim);
}

inline void _llk_unpack_fast_untilize_block_(const std::uint32_t address, const std::uint32_t unit_dim)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    // Reset both SrcA/SrcB Z/W counters before each block so the MOP is
    // stateless across calls. SrcB is included here because fp32 DEST mode uses
    // the same MOP shape with a zero-SrcB dvalid sideband.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;

    semaphore_post(semaphore::UNPACK_SYNC);
    // The base-address store above is a RISCV write to Tensix backend config,
    // and the following MOP expands to UNPACR instructions that consume it.
    // STALL_UNPACK blocks those UNPACRs; TRISC_CFG is STALLWAIT C10, which waits
    // until the current TRISC's config/GPR/TDMA write requests are processed.
    // ISA: BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md,
    // BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    ckernel::ckernel_template::run();

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_bfp_block_(const std::uint32_t address, const std::uint32_t tile_stride_16B, const std::uint32_t unit_dim)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address + tile_stride_16B * (unit_dim - 1)), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(tile_stride_16B > 0, "fast_untilize BFP tile stride must be greater than zero");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize BFP unpack supports unit_dim 2, 3, or 4");

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    const bool use_context_0     = unp_cfg_context == 0;
    const std::uint32_t upk0_reg = use_context_0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;
    cfg[SCRATCH_SEC0_val_ADDR32] = tile_stride_16B;

    semaphore_post(semaphore::UNPACK_SYNC);
    // Orders the RISCV writes to the unpack base address and scratch stride
    // before any UNPACR from the MOP can start. The scratch value is consumed by
    // later CFGSHIFTMASK address updates in this BFP loop.
    // ISA: STALLWAIT C10 plus the RISCV-store-to-Tensix-instruction ordering rule.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Compressed BFP tiles are not laid out as one dense multi-tile payload:
    // each tile has its own exponent section. Run the same per-tile face MOP,
    // then advance the unpack base register by the CB page stride before the
    // next tile. Non-BFP input can use one wider base address for the chunk.
    for (std::uint32_t tile = 0; tile < unit_dim; tile++)
    {
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        ckernel::ckernel_template::run();

        if (tile + 1 < unit_dim)
        {
            if (use_context_0)
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            }
            else
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            }
            // CFGSHIFTMASK updates the unpack base register for the next tile.
            // Keep the scheduling bubble local to the address update so the next
            // loop iteration can issue its UNPACR MOP without reasoning about the
            // two-cycle config write. ISA: CFGSHIFTMASK.md.
            TTI_NOP;
        }
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_uninit_()
{
}

} // namespace ckernel
