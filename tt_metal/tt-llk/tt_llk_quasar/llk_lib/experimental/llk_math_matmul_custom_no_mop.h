// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "llk_math_matmul.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/*************************************************************************
 * LLK MATH MATMUL CUSTOM NO MOP
 *************************************************************************/

// The MOP-based matmul programs BANK0 (see _llk_math_matmul_mop_config_) with
//     outer = 1, inner = FIDELITY_PHASES, LOOP_INSTR0 = REPLAY(0, len), LOOP_INSTR1 = matmul_op,
//     LOOP0_LAST_INSTR = matmul_op_last
// which the MOP expands into [REPLAY, matmul_op] x (FIDELITY_PHASES - 1) then [REPLAY, matmul_op_last].
//
// This path issues that identical instruction stream straight from the RISC core, so MOP BANK0 stays
// free for a fused op that needs it. Same replay image (_llk_math_matmul_load_replay_) and same addrmod
// slots (_llk_math_matmul_addrmod_) as the MOP path, so the numeric result is unchanged, only the
// issue mechanism differs, trading MOP occupancy for RISC instruction-issue bandwidth.

/**
 * @brief Issues the MVMUL stream for one Tile x Tile matrix multiply directly, bypassing the MOP.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, replays the non-DI MXFP4_2x sequence (8 MVMULs per tile instead of 16).
 * @param reuse_a: True when SrcA is held across the reuse dimension (ct_dim >= rt_dim), so the closing MVMUL releases SrcA; otherwise it releases SrcB.
 * @note Call @ref _llk_math_matmul_init_no_mop_ with matching template args first this replays the buffer and addrmod slots it programmed.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_run_no_mop_(const bool reuse_a)
{
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);
    constexpr std::uint32_t replay_buf_len  = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    constexpr std::uint8_t fidelity_phase_completion_addr_mod = ADDR_MOD_4;
    constexpr std::uint8_t tile_completion_addr_mod           = ADDR_MOD_5;

    // load_mode = 0 makes REPLAY issue replay_buffer[0 +: replay_buf_len] to Tensix instead of recording
    // into it, which is what the MOP's LOOP_INSTR0 does. FIDELITY_PHASES is constexpr, so this unrolls.
    for (std::uint32_t phase = 0; phase < FIDELITY_PHASES - 1; phase++)
    {
        TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
        // matmul_op: close this fidelity phase, rewind dest to the start of the tile, advance the fidelity counter.
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, fidelity_phase_completion_addr_mod, 0);
    }

    TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
    // matmul_op_last: close the tile, advance dest to the next one, clear the fidelity counter, and
    // release whichever operand is not being reused across the block row.
    if (reuse_a)
    {
        TTI_MVMUL(p_setrwc::CLR_A, 0, tile_completion_addr_mod, 0);
    }
    else
    {
        TTI_MVMUL(p_setrwc::CLR_B, 0, tile_completion_addr_mod, 0);
    }
}

/**
 * @brief Initializes addrmods and the replay buffer for a matrix multiply that runs without a MOP.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note On the unpack thread, pair with @ref _llk_unpack_matmul_init_ (T0); on the pack thread, with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_matmul_block_no_mop_ runs the configured matmul with matching template args.
 * @note Reload before every matmul that is interleaved with another replay-using op: every Quasar LLK
 *       records its replay buffer at slot 0, so an intervening op overwrites this one's image. Unlike the
 *       MOP path there is no MOP config holding the length, so a stale image would silently replay the
 *       wrong instructions.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_init_no_mop_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
    _llk_math_matmul_load_replay_<ENABLE_2X_FORMAT>();

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA over a block of tiles, without a MOP.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * IMPORTANT NOTES:
 * 1. Dest index always assumed to start at 0 for this operation.
 * 2. If matrix multiplication includes kt_dim > 1 such that matrix multiplication is:
 *    Input 0 [rt_dim, kt_dim] x Input 1 [kt_dim, ct_dim] = Output [rt_dim, ct_dim],
 *    be aware that this function does not iterate over kt_dim; iterate over kt_dim externally to this function.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note Call @ref _llk_math_matmul_init_no_mop_ with matching template args before this function.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_block_no_mop_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // Matmul Block, reset the dest addr to 0 for fused kernels
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // reuse-dim

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            _llk_math_matmul_run_no_mop_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(reuse_a);

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
            }
        }

        //  When rt_dim > ct_dim, the matmul block dest tile indices are not equal to 0,1,2,3..7
        //  Instead they have a ct_dim stride, for instance:
        //  If rt_dim = 4, ct_dim = 2, dest tile indices = 0,2,4,6,  1,3,5,7
        //  If rt_dim = 4, ct_dim = 3, dest tile indices = 0,3,6,9,  1,4,7,10,  2,5,8,11
        //  Below offsets by 1 tile * (t+1), for every subsequence above to start from the next dest_idx
        if (!reuse_a && ct_dim >= 2)
        {
            TT_SETRWC(p_setrwc::CLR_NONE, 0, 64 * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
