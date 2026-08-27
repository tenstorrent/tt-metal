// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "lltt.h"
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
// Replay-buffer slot for the recorded part of the math body, and its exact count:
//   8 SFPMAD + 8 SFPNOP = 16
// The length MUST equal the number of instructions actually emitted by
// _compute_ema_math_body_(), or the replay buffer misaligns and the kernel silently
// executes the wrong sequence. Update both together.
//
// The record is deliberately 16 and not the body's full 19. cmath_common.h splits the
// per-thread replay buffer between the two issuers -- ckernel::math::replay_buf_offset
// == 16, so [0, 16) is the SFPU half and [16, 32) belongs to the FPU. Recording all 19
// here would run to slot 18 and overwrite the first three FPU slots. That is not
// theoretical: _llk_math_transpose_dest_init_ records its MOP into [16, 32) and the
// per-tile op replays out of it, so a kernel that inits transpose_dest and then runs an
// EMA tile gets silently wrong transposes. Measured on n300 with a NoExec probe that
// claims [0, N) between that record and its replays: N <= 16 leaves all 16384 outputs
// correct, N = 17/18/19 corrupts 4096/8192/12288 of them -- exactly 4096 per FPU slot
// crossed.
//
// Keeping the record at 16 and issuing the other three inline costs 0.04 cycles/tile
// (247.89 -> 247.93) and 84 bytes. See _compute_ema_math_() below.
//
// Within the SFPU half, slot 0 is shared: the accurate exp path records [0, 32) and topk
// up to [0, 29), so a caller interleaving EMA with either must not rely on slot 0
// surviving across the call.
constexpr std::uint32_t EMA_MATH_REPLAY_SLOT = 0;
constexpr std::uint32_t EMA_MATH_REPLAY_LEN  = 16;

// The recorded part: exactly the 8 SFPMAD + 8 SFPNOP pairs. The two SFPTRANSPs and the
// SFPMOV that bracket them are issued inline by _compute_ema_math_() instead, to hold the
// record to 16 slots.
sfpi_inline void _compute_ema_math_body_()
{
    // EMA equation: EMA_new = alpha * EMA_old + beta * input
    // Registers: LREG0-3 = the 4 input rows, updated in place to become the 4 outputs;
    //            LREG4 = EMA_old carried in (and the new carry out); LREG5 = alpha;
    //            LREG6 = beta; LREG7 = temp holding alpha * previous row.
    //
    // Two SFPMADs per row: LREG7 = alpha*prev, then row = beta*input + LREG7. Both sit on
    // the dependency chain, so each needs an SFPNOP behind it (2-cycle SFPMAD write
    // latency). Those NOPs cannot be filled: the rows must occupy LREG0-3 to be transposed
    // as a group, they are live for the whole span where the slots exist, and the stores
    // need the post-transpose layout, so no independent work can be scheduled into them.
    //
    // Reassociating to remove them (pre-scaling the inputs by beta) does work and is worth
    // a further -19%, but it changes which product gets its own rounding and is therefore
    // not bit-neutral: measured over a 1000-alpha sweep it moves 87 of 2048000 outputs, 84
    // of them by one bfloat16 ULP, at unchanged accuracy. It was not taken here. Replaying
    // the body instead recovers nearly the same win with no numerical change at all -- see
    // _compute_ema_math_() below.

    // Step 1(in0): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG4 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in0): Calculate final EMA = β * in0 + α * EMA_old
    // LREG0 = (LREG6 * LREG0) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG0 (2-cycle operation)

    // Step 1(in1): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG0 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in1): Calculate final EMA = β * in1 + α * EMA_old
    // LREG1 = (LREG6 * LREG1) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG1 (2-cycle operation)

    // Step 1(in2): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG1 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in2): Calculate final EMA = β * in2 + α * EMA_old
    // LREG2 = (LREG6 * LREG2) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG2 (2-cycle operation)

    // Step 1(in3): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG2 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in3): Calculate final EMA = β * in3 + α * EMA_old
    // LREG3 = (LREG6 * LREG3) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG3 (2-cycle operation)
}

/**
 * @brief Record the math body on the first block of a tile, replay it on the other seven.
 *
 * _compute_ema_math_body_() takes no template parameters, so all 8 blocks in a tile emit
 * byte-identical instructions -- only the surrounding loads and stores differ, through
 * compile-time dst offsets. That makes the body a clean record-once/replay candidate. The
 * 8 x 16 recorded instructions collapse to 16 + 7; the SFPTRANSP/SFPMOV/SFPTRANSP that
 * bracket the body stay inline and are issued on all 8 blocks.
 *
 * @note This is a cycle win, not just a code-size win, and the size of it was surprising:
 *       MATH_ISOLATE drops from 320.77 to 247.93 cycles/tile (-22.7%) on Wormhole n300,
 *       with TEXT_SIZE 3015 -> 2599. The arithmetic is untouched, so the result is
 *       bit-identical -- verified over a 1000-alpha sweep, 2048000 outputs, zero differing.
 *
 * @note Do not read the win as "fewer instructions issued is proportionally faster". The
 *       obvious model -- the TRISC is push-rate bound, so each collapsed instruction buys
 *       time -- fits the headline number (126 instructions collapsed, 72.9 cycles saved,
 *       ~0.58 cycles each) but is contradicted by direct measurement. Moving 3 of the 19
 *       instructions out of the record and back inline puts 21 pushes per tile back, which
 *       that model prices at ~12 cycles; measured, it costs 0.04 (247.89 with all 19
 *       recorded, 247.93 with 16). Both figures reproduce exactly across runs.
 *
 *       What fits: the fully inlined kernel, at 152 pushes per tile, really is push-bound,
 *       which is where the 72.9 cycles come from. Once the body is replayed the bottleneck
 *       moves to replay expansion inside Tensix, and a few extra pushes per block hide
 *       underneath it. So the first ~126 collapsed instructions are nearly all of the win
 *       and the last few are worth almost nothing -- which is what makes holding the record
 *       to the SFPU half of the buffer essentially free.
 */
template <bool RECORD>
sfpi_inline void _compute_ema_math_()
{
    // Transpose the input data to the correct order. Inline, not recorded -- see
    // EMA_MATH_REPLAY_LEN for why the record stops at 16.
    TTI_SFPTRANSP(0, 0, 0, 0);

    if constexpr (RECORD)
    {
        // lltt::Exec so the first block both records and executes; the recorded copy is
        // what the remaining seven replay.
        lltt::record<lltt::Exec>(EMA_MATH_REPLAY_SLOT, EMA_MATH_REPLAY_LEN);
        _compute_ema_math_body_();
    }
    else
    {
        lltt::replay(EMA_MATH_REPLAY_SLOT, EMA_MATH_REPLAY_LEN);
    }

    // Update EMA_old for next iteration: LREG4 = LREG3. Inline, not recorded.
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);

    // Transpose the output data to the correct order. Inline, not recorded.
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
template <std::uint32_t I, std::uint32_t J, bool RECORD = false>
sfpi_inline void _process_ema_block_()
{
    _ema_load_current_input_<I, J>();
    _compute_ema_math_<RECORD>();
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
    _process_ema_block_<0, 0, /* RECORD */ true>();

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
