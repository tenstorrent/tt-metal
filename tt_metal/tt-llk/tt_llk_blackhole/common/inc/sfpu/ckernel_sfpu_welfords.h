// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "api/numeric/bfloat16.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

/**
 * @brief Loads the reciprocal of (idx + 1) into LREG7, using a lookup table if available.
 *
 * This function either loads a precomputed reciprocal value from the provided lookup table
 * (reciprocal_lut) into the LREG7 register, or, if the lookup table entry is not available,
 * computes the reciprocal at runtime as 1.0f/(idx + 1) and loads its bit representation
 * into the register.
 *
 * @tparam reciprocal_size The number of entries in the reciprocal lookup table.
 * @param idx The (zero-based) index (in the reciprocal lookup table) of the value to load.
 * @param reciprocal_lut Lookup table containing precomputed reciprocals packed as uint32_t.
 *
 * @note The reciprocal is written to ckernel::p_sfpu::LREG7.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _load_recip_of_idx_(
    const std::uint32_t idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    if constexpr (reciprocal_size > 0)
    {
        const auto reciprocal = reciprocal_lut[idx];

        TT_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_UPPER,
            reciprocal >> 16);

        TT_SFPLOADI(
            ckernel::p_sfpu::LREG7,
            sfpi::SFPLOADI_MOD0_LOWER,
            reciprocal & 0xFFFF);

        return;
    }

    // Fallback to float division
    const float reciprocal = 1.0f / static_cast<float>(idx + 1);
    const FloatBits reciprocal_bits(reciprocal);

    TT_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_UPPER,
        reciprocal_bits.high16);

    TT_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_LOWER,
        reciprocal_bits.low16);
}

/*
 * @brief Loads the current input at row (I * 4) + J from the current tile in dst reg at offset 0.
 *
 * @tparam I
 * @tparam J
 * 4 inputs each from 32 columns at the current offset are loaded into the LREG0-3 registers
 * respectively from the current tile in dst 0.
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _welfords_load_block_()
{
    constexpr std::uint32_t tile_offset    = 0;
    constexpr std::uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);

    constexpr std::uint32_t offset0 = dst_reg_offset;
    constexpr std::uint32_t offset1 = dst_reg_offset + 2;
    constexpr std::uint32_t offset2 = dst_reg_offset + 16;
    constexpr std::uint32_t offset3 = dst_reg_offset + 18;

    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        offset0);

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG1,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        offset1);

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        offset2);

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        offset3);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

/**
 * @brief Clears the previous mean and m2/variance accumulators.
 *
 * LREG4 is used for the running sum during the first pass.
 * LREG5 is used for the running squared-difference sum during the second pass.
 */
namespace ckernel
{
namespace sfpu
{

sfpi_inline void _clear_previous_mean_and_m2_()
{
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOADI_MOD0_FLOATB,
        0);

    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_FLOATB,
        0);
}

/**
 * @brief Computes one row for the first pass of the two-pass algorithm.
 *
 * The first pass accumulates:
 *
 *     sum(x)
 *
 * in LREG4.
 *
 * The actual mean is calculated after all rows have been processed:
 *
 *     mean = sum(x) / N
 *
 * @tparam input_lreg Input row register. LREG0-LREG3.
 */
template <std::uint32_t input_lreg>
sfpi_inline void _compute_two_pass_mean_row_()
{
    /*
     * LREG6 = input_lreg
     *
     * We use LREG6 as a temporary so that the input register itself
     * remains available for the rest of the block.
     */
    TTI_SFPMOV(
        0,
        input_lreg,
        ckernel::p_sfpu::LREG6,
        0);

    /*
     * LREG4 = LREG4 + LREG6
     *
     * LREG4 contains the running sum.
     */
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG4,
        0);
}

/**
 * @brief Computes one row for the second pass of the two-pass algorithm.
 *
 * The second pass calculates:
 *
 *     (x - mean)^2
 *
 * and accumulates the result in LREG5.
 *
 * The mean is expected to already be present in LREG4.
 *
 * @tparam input_lreg Input row register. LREG0-LREG3.
 */
template <std::uint32_t input_lreg>
sfpi_inline void _compute_two_pass_variance_row_()
{
    /*
     * LREG6 = x - mean
     *
     * LREG11 contains -1.
     */
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG11,
        ckernel::p_sfpu::LREG4,
        input_lreg,
        ckernel::p_sfpu::LREG6,
        0);

    /*
     * LREG5 = LREG5 + (x - mean)^2
     */
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG5,
        0);
}

/**
 * @brief The number of instructions required for one row in the first pass.
 *
 * If _compute_two_pass_mean_row_ is modified, this value must be updated.
 */
constexpr std::uint32_t TWO_PASS_MEAN_INSTR_PER_ROW = 2;

/**
 * @brief The number of instructions required for one row in the second pass.
 *
 * If _compute_two_pass_variance_row_ is modified, this value must be updated.
 */
constexpr std::uint32_t TWO_PASS_VARIANCE_INSTR_PER_ROW = 2;

/**
 * @brief Programs the replay buffer for the first-pass mean computation.
 *
 * The replay buffer contains the operation for each of the four input LREGs.
 */
sfpi_inline void _program_two_pass_mean_replay_buffer_()
{
    lltt::record(0, TWO_PASS_MEAN_INSTR_PER_ROW * 4);

    _compute_two_pass_mean_row_<ckernel::p_sfpu::LREG0>();
    _compute_two_pass_mean_row_<ckernel::p_sfpu::LREG1>();
    _compute_two_pass_mean_row_<ckernel::p_sfpu::LREG2>();
    _compute_two_pass_mean_row_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Programs the replay buffer for the second-pass variance computation.
 */
sfpi_inline void _program_two_pass_variance_replay_buffer_()
{
    lltt::record(0, TWO_PASS_VARIANCE_INSTR_PER_ROW * 4);

    _compute_two_pass_variance_row_<ckernel::p_sfpu::LREG0>();
    _compute_two_pass_variance_row_<ckernel::p_sfpu::LREG1>();
    _compute_two_pass_variance_row_<ckernel::p_sfpu::LREG2>();
    _compute_two_pass_variance_row_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Executes the replay buffer for the first-pass mean computation.
 *
 * @tparam input_lreg The input LREG to replay. LREG0-LREG3.
 */
template <std::uint32_t input_lreg>
sfpi_inline void _execute_two_pass_mean_replay_buffer_()
{
    lltt::replay(
        TWO_PASS_MEAN_INSTR_PER_ROW * input_lreg,
        TWO_PASS_MEAN_INSTR_PER_ROW);
}

/**
 * @brief Executes the replay buffer for the second-pass variance computation.
 *
 * @tparam input_lreg The input LREG to replay. LREG0-LREG3.
 */
template <std::uint32_t input_lreg>
sfpi_inline void _execute_two_pass_variance_replay_buffer_()
{
    lltt::replay(
        TWO_PASS_VARIANCE_INSTR_PER_ROW * input_lreg,
        TWO_PASS_VARIANCE_INSTR_PER_ROW);
}

/**
 * @brief Calculates the first-pass sum for a single block of 4 rows and 32 columns.
 *
 * Each block contains four rows. The values are accumulated into LREG4.
 *
 * @tparam I
 * @tparam J
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _calculate_two_pass_mean_block_()
{
    _welfords_load_block_<I, J>();

    _execute_two_pass_mean_replay_buffer_<ckernel::p_sfpu::LREG0>();
    _execute_two_pass_mean_replay_buffer_<ckernel::p_sfpu::LREG1>();
    _execute_two_pass_mean_replay_buffer_<ckernel::p_sfpu::LREG2>();
    _execute_two_pass_mean_replay_buffer_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Calculates the second-pass squared-difference sum for a single block.
 *
 * LREG4 must contain the mean before this function is called.
 *
 * @tparam I
 * @tparam J
 */
template <std::uint32_t I, std::uint32_t J>
sfpi_inline void _calculate_two_pass_variance_block_()
{
    _welfords_load_block_<I, J>();

    _execute_two_pass_variance_replay_buffer_<ckernel::p_sfpu::LREG0>();
    _execute_two_pass_variance_replay_buffer_<ckernel::p_sfpu::LREG1>();
    _execute_two_pass_variance_replay_buffer_<ckernel::p_sfpu::LREG2>();
    _execute_two_pass_variance_replay_buffer_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Calculates the mean for a full 32x32 tile using the first pass.
 *
 * The tile contains 32 rows, with 32 values in each row.
 *
 * LREG4 contains the accumulated sum when the function begins.
 * After this function completes, LREG4 contains the mean.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param reciprocal_lut Reciprocal lookup table.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_two_pass_mean_tile_(
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _calculate_two_pass_mean_block_<0, 0>();
    _calculate_two_pass_mean_block_<0, 1>();
    _calculate_two_pass_mean_block_<0, 2>();
    _calculate_two_pass_mean_block_<0, 3>();

    _calculate_two_pass_mean_block_<1, 0>();
    _calculate_two_pass_mean_block_<1, 1>();
    _calculate_two_pass_mean_block_<1, 2>();
    _calculate_two_pass_mean_block_<1, 3>();

    /*
     * There are 32 rows in the tile.
     *
     * reciprocal_lut[31] = 1 / 32
     *
     * Convert:
     *
     *     sum(x) -> mean(x)
     */
    _load_recip_of_idx_<reciprocal_size>(31, reciprocal_lut);

    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        0);
}

/**
 * @brief Calculates the variance for a full 32x32 tile using the second pass.
 *
 * LREG4 must contain the mean calculated by the first pass.
 * LREG5 contains the accumulated squared differences after this function.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param reciprocal_lut Reciprocal lookup table.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_two_pass_variance_tile_(
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _calculate_two_pass_variance_block_<0, 0>();
    _calculate_two_pass_variance_block_<0, 1>();
    _calculate_two_pass_variance_block_<0, 2>();
    _calculate_two_pass_variance_block_<0, 3>();

    _calculate_two_pass_variance_block_<1, 0>();
    _calculate_two_pass_variance_block_<1, 1>();
    _calculate_two_pass_variance_block_<1, 2>();
    _calculate_two_pass_variance_block_<1, 3>();

    /*
     * Convert:
     *
     *     sum((x - mean)^2) -> variance
     *
     * There are 32 rows in the tile.
     */
    _load_recip_of_idx_<reciprocal_size>(31, reciprocal_lut);

    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);
}

/**
 * @brief Calculates the mean and variance for a full tile using two passes.
 *
 * This replaces the Welford recurrence:
 *
 *     pass 1: sum(x)
 *     mean = sum(x) / N
 *     pass 2: sum((x - mean)^2)
 *     variance = sum((x - mean)^2) / N
 *
 * The resulting mean and variance are stored in LREG4 and LREG5.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param reciprocal_lut Reciprocal lookup table.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_two_pass_tile_(
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _clear_previous_mean_and_m2_();

    /*
     * First pass:
     *
     * LREG4 = sum(x)
     */
    _calculate_two_pass_mean_tile_(reciprocal_lut);

    /*
     * Second pass:
     *
     * LREG5 = sum((x - mean)^2)
     */
    _calculate_two_pass_variance_tile_(reciprocal_lut);
}

/*
 * @brief Calculates the two-pass algorithm for a tile in the dst reg on a subset of rows.
 *
 * This function calculates the mean and variance using two passes over the selected rows.
 *
 * It assumes that the current input tile is present in dst 0.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param start_idx The index of the first processed row.
 * @param start_row The offset of the row to start from.
 * @param num_rows The number of rows to process.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_two_pass_partial_tile_(
    std::uint32_t start_idx,
    std::uint32_t start_row,
    std::uint32_t num_rows,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    if (num_rows == 0)
    {
        return;
    }

    const std::uint32_t end_row = start_row + num_rows;

    /*
     * ------------------------------------------------------------------------
     * First pass: accumulate sum(x)
     * ------------------------------------------------------------------------
     */
    _clear_previous_mean_and_m2_();

    std::uint32_t mean_idx = start_idx;

    _calculate_two_pass_mean_block_partial_<0, 0>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<0, 1>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<0, 2>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<0, 3>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<1, 0>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<1, 1>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<1, 2>(
        mean_idx, start_row, end_row);

    _calculate_two_pass_mean_block_partial_<1, 3>(
        mean_idx, start_row, end_row);

    /*
     * Convert sum(x) into mean.
     *
     * reciprocal_lut[num_rows - 1] = 1 / num_rows
     */
    _load_recip_of_idx_<reciprocal_size>(
        num_rows - 1,
        reciprocal_lut);

    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        0);

    /*
     * ------------------------------------------------------------------------
     * Second pass: accumulate (x - mean)^2
     * ------------------------------------------------------------------------
     */
    _clear_two_pass_variance_accumulator_();

    std::uint32_t variance_idx = start_idx;

    _calculate_two_pass_variance_block_partial_<0, 0>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<0, 1>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<0, 2>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<0, 3>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<1, 0>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<1, 1>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<1, 2>(
        variance_idx, start_row, end_row);

    _calculate_two_pass_variance_block_partial_<1, 3>(
        variance_idx, start_row, end_row);

    /*
     * Convert the accumulated squared differences to population variance.
     */
    _load_recip_of_idx_<reciprocal_size>(
        num_rows - 1,
        reciprocal_lut);

    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);
}

/**
 * @brief Stores the mean and m2 values to the tile in the dst reg at offset 0 and 1 respectively.
 *
 * This function stores the mean and m2 values to the tile in the dst reg at offset 0 and 1
 * respectively. The values are stored in "raw" format. i.e. a total of 32 mean values are stored in
 * the first face of the tile in dst at offset 0. These values are stored at even indices only.
 * Thus, the values take up 4 rows in the first face with 8 values per row. The m2 values are stored
 * in the same way but at tile offset 1.
 */
sfpi_inline void _store_mean_m2_to_dst_()
{
    constexpr std::uint32_t mean_tile_offset = 0;
    constexpr std::uint32_t m2_tile_offset   = 64;

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        m2_tile_offset);
}

/*
 * @brief Stores the mean and m2 values to the tile in the dst reg at offset 0 and 1 respectively
 *        for a given group.
 *
 * This function does the same as _store_mean_m2_to_dst_ but allows for the data to be stored at an
 * offset that is dependent on the group id. This allows for data of multiple groups to be stored in
 * the same tile.
 * @note Since group_id is known at runtime, we use TT_SFPSTORE instead of TTI_SFPSTORE.
 * @param group_id The group id to store the data for.
 */
sfpi_inline void _store_mean_m2_to_dst_group_(std::uint32_t group_id)
{
    constexpr std::uint32_t mean_tile_offset = 0;
    constexpr std::uint32_t m2_tile_offset   = 64;

    TT_SFPSTORE(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + (group_id << 2));

    TT_SFPSTORE(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        m2_tile_offset + (group_id << 2));
}

/*
 * @brief Loads the mean and m2 values from the tile in the dst reg into LREGs.
 *
 * This function loads the mean and m2 values from the tiles in the dst reg into LREGs.
 * The mean is loaded into LREG4, and the m2 is loaded into LREG5.
 * It assumes that the mean and m2 values were stored by _store_mean_m2_to_dst_
 * (i.e., they are each stored in the "raw" format).
 */
sfpi_inline void _load_mean_m2_from_dst_()
{
    constexpr std::uint32_t mean_tile_offset = 0;
    constexpr std::uint32_t m2_tile_offset   = 64;

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset);

    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        m2_tile_offset);
}

/*
 * @brief Loads the mean and m2 values from the tile in the dst reg into LREGs for a given group.
 *
 * This function does the same as _load_mean_m2_from_dst_ but allows for the data to be loaded from
 * an offset that is dependent on the group id. This allows for data of multiple groups to be loaded
 * from the same tile.
 * @note Since group_id is known at runtime, we use TT_SFPLOAD instead of TTI_SFPLOAD.
 * @param group_id The group id to load the data for.
 */
sfpi_inline void _load_mean_m2_from_dst_group_(std::uint32_t group_id)
{
    constexpr std::uint32_t mean_tile_offset = 0;
    constexpr std::uint32_t m2_tile_offset   = 64;

    TT_SFPLOAD(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + (group_id << 2));

    TT_SFPLOAD(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOAD_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        m2_tile_offset + (group_id << 2));
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg.
 *
 * This function stores the mean and variance values to the tile in the dst reg.
 * It assumes that the mean and m2 values are placed in LREG4 and LREG5, respectively.
 * These values are placed in the first row of the tile in dst. The reciprocal LUT, if provided,
 * is used to load the reciprocal of the sample count.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_row_(
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(
        scale_idx,
        reciprocal_lut);

    // Move mean to LREG0
    TTI_SFPMOV(
        0,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        0);

    TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0);

    // Convert M2 to variance and move to LREG4
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        0);

    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, 0, 0);

    // Move all the values to a single row
    TTI_SFPTRANSP(0, 0, 0, 0);

    constexpr std::uint32_t offset0 = 0;
    constexpr std::uint32_t offset1 = 2;
    constexpr std::uint32_t offset2 = 16;
    constexpr std::uint32_t offset3 = 18;

    constexpr std::uint32_t mean_tile_offset = 0;

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + offset0);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG1,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + offset1);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + offset2);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + offset3);

    constexpr std::uint32_t var_tile_offset = 64;

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        var_tile_offset + offset0);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        var_tile_offset + offset1);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        var_tile_offset + offset2);

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPSTORE_MOD0_FMT_SRCB,
        ckernel::ADDR_MOD_7,
        var_tile_offset + offset3);
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg.
 *
 * This function stores the mean and variance values to the tile in the dst reg.
 * It assumes that the mean and m2 values are placed in LREG4 and LREG5, respectively.
 * These values are placed in the first face of the tile in dst. The reciprocal LUT, if provided,
 * is used to load the reciprocal of the sample count.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_raw_(
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(
        scale_idx,
        reciprocal_lut);

    // Convert M2 to variance in LREG5
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);

    constexpr std::uint32_t mean_tile_offset = 0;

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG4,
        0,
        ckernel::ADDR_MOD_7,
        mean_tile_offset);

    constexpr std::uint32_t var_tile_offset = 64;

    TTI_SFPSTORE(
        ckernel::p_sfpu::LREG5,
        0,
        ckernel::ADDR_MOD_7,
        var_tile_offset);
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg for a given group.
 *
 * This function does the same as _store_mean_var_to_dst_raw_ but allows for the data to be stored
 * at an offset that is dependent on the group id.
 * @note Since group_id is known at runtime, we use TT_SFPSTORE instead of TTI_SFPSTORE.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param group_id The group id to store the data for.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_raw_group_(
    std::uint32_t group_id,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(
        scale_idx,
        reciprocal_lut);

    // Convert M2 to variance in LREG5
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);

    constexpr std::uint32_t mean_tile_offset = 0;

    TT_SFPSTORE(
        ckernel::p_sfpu::LREG4,
        0,
        ckernel::ADDR_MOD_7,
        mean_tile_offset + (group_id << 2));

    constexpr std::uint32_t var_tile_offset = 64;

    TT_SFPSTORE(
        ckernel::p_sfpu::LREG5,
        0,
        ckernel::ADDR_MOD_7,
        var_tile_offset + (group_id << 2));
}

} // namespace sfpu
} // namespace ckernel