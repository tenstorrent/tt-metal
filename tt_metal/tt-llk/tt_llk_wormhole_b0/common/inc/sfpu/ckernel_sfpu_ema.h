// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "sfpi.h"

/*
 * @brief Loads the current input at row (I * 4) + J from the current tile in dst 0.
 *
 * @tparam I
 * @tparam J
 *
 * 4 inputs each from 32 columns at the current offset are loaded into the LREG0-3 registers
 * respectively from the current tile in dst 0.
 * Values are expected to be bfloat16 format.
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _ema_load_current_input_()
{
    constexpr std::uint32_t tile_offset    = 0; // offset for tile 0 in dst
    constexpr std::uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr std::uint32_t offset0        = dst_reg_offset;
    constexpr std::uint32_t offset1        = dst_reg_offset + 2;
    constexpr std::uint32_t offset2        = dst_reg_offset + 16;
    constexpr std::uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset0); // row0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset1); // row1
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset2); // row2
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset3); // row3
}

/**
 * @brief Stores the current input at row (I * 4) + J to the corresponding position in tile 1.
 *
 * @tparam I
 * @tparam J
 *
 * 4 inputs each from 32 columns at the current offset are stored from the LREG0-3 registers
 * respectively to tile 1.
 * Values are stored in bfloat16 format.
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _ema_store_current_input_()
{
    constexpr std::uint32_t tile_offset    = 64; // offset for tile 1 in dst
    constexpr std::uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr std::uint32_t offset0        = dst_reg_offset;
    constexpr std::uint32_t offset1        = dst_reg_offset + 2;
    constexpr std::uint32_t offset2        = dst_reg_offset + 16;
    constexpr std::uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset0); // row0
    TTI_SFPSTORE(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset1); // row1
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset2); // row2
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset3); // row3
}

/*
 * @brief Computes the exponential moving average for 4 inputs each from 32 columns.
 *
 * The function computes the exponential moving average for 4 inputs each from 32 columns.
 * It operates on the input data in the LREG0-3 registers, updating the EMA values using the loaded
 * alpha and beta parameters. It takes the previous EMA value from LREG4 for all 32 columns.
 * The output is stored in the LREG0-3 registers.
 */
sfpi_inline void _compute_ema_math_()
{
    // Transpose the input data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);

    // EMA_new = alpha * EMA_old + beta * input, chained across the 4 rows.
    //
    // Registers: LREG0-3 hold the 4 input rows and are updated in place to become the 4
    // outputs; LREG4 carries EMA_old in from the previous block and the new carry out;
    // LREG5 = alpha; LREG6 = beta. LREG7 is not used -- the earlier schedule needed it as
    // a temp, this one does not.
    //
    // That earlier schedule (no longer what the code below does) spent two MADs per row:
    // LREG7 = alpha * prev, then row = beta * input + LREG7. Both halves sat on the
    // dependency chain, so each needed an SFPNOP behind it and the block cost 8 MADs plus
    // 8 NOPs.
    //
    // Scaling every input by beta up front instead leaves one MAD per row on the chain
    // (row_i = alpha * row_{i-1} + beta_scaled_i, a single fused multiply-add), and the
    // four scaling multiplies are mutually independent, so they can be dealt into the
    // chain's latency slots rather than stalling behind it. Same algebra, half the
    // chain: 8 MADs and 2 NOPs instead of 8 MADs and 8 NOPs.
    //
    // It is NOT the same floating-point arithmetic, and this is not a reordering of
    // independent instructions -- it is a reassociation. SFPMAD performs exactly one
    // rounding per instruction (see WormholeB0 SFPMAD.md: partially fused, single
    // rounding, and "adding zero ... equivalent to a standalone multiply"), so both forms
    // round the same number of times but round *different quantities*: the old form gave
    // alpha*prev its own rounding and partially fused beta*input, this one gives beta*input
    // its own rounding and partially fuses alpha*prev.
    //
    // Consequences, measured on Wormhole n300 over a 1000-alpha sweep (alpha = k/1000,
    // 2048 outputs each, run on both kernels -- test_sfpu_ema_alpha_sweep.py):
    //
    //   * Accuracy is unchanged. Against an fp64 reference the two forms have a mean RMS
    //     error ratio of 1.000001 and an identical worst peak error; no alpha differs by
    //     more than 2%. Neither form is systematically closer to the true recurrence.
    //   * Results are NOT bit-identical in general. 923 of 1000 alphas came out identical,
    //     but 87 of 2048000 outputs differ, 84 of them by exactly one bfloat16 ULP. Only
    //     the alphas that are exact binary fractions are structurally safe -- there
    //     alpha*prev cannot round, so there is nothing to reassociate.
    //
    // Do not be tempted by the argument that an fp32 perturbation of ~2^-24 cannot survive
    // a bfloat16 store at 2^-9. It is false: a perturbation far below one ULP still flips
    // the rounded result whenever the exact value sits near a rounding midpoint. It makes
    // disagreement rare, not impossible. An earlier version of this comment claimed
    // otherwise on exactly that reasoning, and the sweep above disproves it.
    //
    // Pre-scale in0/in1 by beta. Independent of everything below.
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);

    // in0: LREG0 = alpha * EMA_old + beta * in0
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    // Pre-scale in2, covering the write latency of the MAD above.
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0);

    // in1: LREG1 = alpha * LREG0 + beta * in1
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    // Pre-scale in3, covering the write latency of the MAD above.
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 0);

    // in2: LREG2 = alpha * LREG1 + beta * in2
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP; // no independent work left to cover LREG2

    // in3: LREG3 = alpha * LREG2 + beta * in3
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPNOP; // SFPMOV below reads LREG3

    // Update EMA_old for next iteration
    // LREG4 = LREG3 (copy new EMA to old EMA register)
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);

    // Transpose the output data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);
}

/**
 * @brief Processes a single EMA block (load inputs, compute EMA, store results).
 *
 * @tparam I Row group index (0-1)
 * @tparam J Column group index (0-3)
 *
 * This is a helper function that performs all three steps for a single block:
 * load inputs, compute EMA, and store results.
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _process_ema_block_()
{
    _ema_load_current_input_<I, J>();
    _compute_ema_math_();
    _ema_store_current_input_<I, J>();
}

namespace ckernel
{
namespace sfpu
{
/**
 * @brief Loads the alpha and beta values into the corresponding SFPU registers.
 *
 * @param alpha The alpha parameter, typically the smoothing factor for the EMA, in 32-bit format.
 * @param beta  The beta parameter, in 32-bit format, representing (1 - alpha) or a similar value.
 *
 * The values dictate the amount of weight given to the previous output and the current input.
 * It follows the formula: EMA_new = α * EMA_old + β * input
 * These values are loaded into the LREG5 (α) and LREG6 (β) registers.
 * The 32 bit values are expected to be the float32 representation of the alpha and beta values.
 */
sfpi_inline void _load_alpha_beta_(std::uint32_t alpha, std::uint32_t beta)
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, alpha >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, alpha & 0xFFFF);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, beta >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, beta & 0xFFFF);
}

/**
 * @brief Clears the previous EMA output stored in the designated register (LREG4).
 *
 * This function zeroes out the register (LREG4) used for storing the previous EMA value,
 * preparing for a new calculation cycle. Typically invoked at the beginning of the
 * calculation for a new EMA sequence.
 */
sfpi_inline void _clear_previous_output_()
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
}

/**
 * @brief Calculates the Exponential Moving Average (EMA) for the input tile in dst 0.
 *
 * Executes the EMA update for all input fragments in the tile in dst 0.
 * The function processes the input data in a loop, updating the EMA values using the loaded
 * alpha and beta parameters. It operates on 32 inputs each from 32 columns, present in a
 * 32x32 tile in dst 0. It also takes the previous EMA value from LREG4 for all 32 columns.
 * The output is stored in the corresponding position in tile 1. The last output for all 32
 * columns is also held in LREG4 for use by the next tile.
 */
sfpi_inline void _calculate_ema_tile_()
{
    // Transpose the input data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);

    // We load 4 rows of a tile (with 32 columns each) at a time and process them.
    // To finish the entire tile, we need to repeat this process 8 times.

    // Process the first block (4 rows of 32 columns)
    _process_ema_block_<0, 0>();

    // Repeat this 7 more times to process the remaining blocks
    _process_ema_block_<0, 1>();
    _process_ema_block_<0, 2>();
    _process_ema_block_<0, 3>();

    _process_ema_block_<1, 0>();
    _process_ema_block_<1, 1>();
    _process_ema_block_<1, 2>();
    _process_ema_block_<1, 3>();

    // Transpose the output data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);
}
} // namespace sfpu
} // namespace ckernel
