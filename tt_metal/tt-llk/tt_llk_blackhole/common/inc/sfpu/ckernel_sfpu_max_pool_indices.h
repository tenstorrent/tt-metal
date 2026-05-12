// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 *        Can reduce up to 9 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to perform (unused).
 * @tparam layout Data layout format, either TILE (default) or ROW_MAJOR.
 * @tparam accumulate Whether to accumulate results for large kernels (default is false).
 * @param values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @param indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param chunk The chunk index for large kernel accumulation.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS             = 8,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE,
    bool accumulate            = false>
inline void _calculate_max_pool_with_indices_(const std::uint32_t values_tile_idx, const std::uint32_t indices_tile_idx, const std::uint32_t chunk)
{
    // size of each tile in Dest is 64 rows
    constexpr std::uint32_t dst_tile_size   = 64;
    const std::uint32_t values_tile_offset  = values_tile_idx * dst_tile_size;
    const std::uint32_t indices_tile_offset = indices_tile_idx * dst_tile_size;
    // each face is 16 rows
    constexpr std::uint32_t face_offset    = 16;
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // ROW MAJOR DATA VERSION OF MPWI
        // DATA IS EXPECTED TO BE IN THE FOLLOWING ORDER IN DEST:
        // Face 0 Row 0
        // Face 1 Row 0
        // Face 0 Row 1
        // Face 1 Row 1
        // Face 0 Row 2
        // Face 1 Row 2
        // Face 0 Row 3
        // Face 1 Row 3
        // Face 0 Row 4
        // Face 1 Row 4
        // Face 0 Row 5
        // Face 1 Row 5
        // Face 0 Row 6
        // Face 1 Row 6
        // Face 0 Row 7
        // Face 1 Row 7
        // Face 0 Row 8
        // Face 1 Row 8

        auto process_columns = [values_tile_offset, indices_tile_offset](const std::uint32_t col_offset) __attribute__((always_inline))
        {
            // data
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 12 + col_offset);
            // index
            TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 12 + col_offset);

            // sort 4 rows
            lltt::replay(0, 7);

            // data
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 16 + col_offset);
            // index
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 16 + col_offset);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 0 + col_offset);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 0 + col_offset);
        };

        // F0 + F1 even and odd cols
        constexpr int even_column_offset = 0;
        constexpr int odd_column_offset  = 2;
        process_columns(even_column_offset);
        process_columns(odd_column_offset);
    }
    else
    {
        static_assert(!accumulate, "accumulate mode is not supported for TILE layout");
        // TILE (ORIGINAL) VERSION OF MPWI
        // F0
        // data
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 0); // even cols
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 4);
        TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 2); // odd cols
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 6);
        // index
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 0); // even cols
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 4);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 2); // odd cols
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 6);

        // sort 4 rows
        lltt::replay(0, 7);

        // data
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 8);  // even cols
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 10); // odd cols
        // index
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 8);  // even cols
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 10); // odd cols

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 0);
        TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + 2);
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_tile_offset + 2);

        // F1
        // data
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset); // even cols
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 4);
        TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 2); // odd cols
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 6);
        // index
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset); // even cols
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 4);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 2); // odd cols
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 6);

        // sort 4 rows
        lltt::replay(0, 7);

        // data
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 8);  // even cols
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 10); // odd cols
        // index
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 8);  // even cols
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 10); // odd cols

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset);
        TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, values_tile_offset + face_offset + 2);
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_tile_offset + face_offset + 2);
    }
}

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 *        Can reduce up to 32 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to use for the MaxPool operation (unused).
 * @tparam accumulate Whether to accumulate results for large kernels (default is false).
 * @param values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @param indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param chunk The chunk index for large kernel accumulation.
 *
 * Note this function is only implemented for ROW_MAJOR data layout, so when _init_max_pool_with_indices_ is called
 * it must be called with layout=DataLayout::ROW_MAJOR.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS, bool accumulate = false>
inline void _calculate_max_pool_with_indices_generic_(const std::uint32_t values_tile_idx, const std::uint32_t indices_tile_idx, const std::uint32_t chunk)
{
    // size of each tile in Dest is 64 rows
    constexpr std::uint32_t dst_tile_size         = 64;
    const std::uint32_t values_tile_offset        = values_tile_idx * dst_tile_size;
    const std::uint32_t indices_tile_offset       = indices_tile_idx * dst_tile_size;
    const std::uint32_t values_accum_tile_offset  = (values_tile_idx + 1) * dst_tile_size;
    const std::uint32_t indices_accum_tile_offset = (indices_tile_idx + 1) * dst_tile_size;
    // each face is 16 rows
    constexpr std::uint32_t eight_row_offset   = 16;
    constexpr std::uint32_t sixteen_row_offset = 32;
    constexpr std::uint8_t instr_mod_index     = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // ROW MAJOR DATA VERSION OF MPWI
    // DATA IS EXPECTED TO BE IN THE FOLLOWING ORDER IN DEST:
    // Face 0 Row 0
    // Face 1 Row 0
    // Face 0 Row 1
    // Face 1 Row 1
    // .
    // .
    // .
    // Face 0 Row 31
    // Face 1 Row 31

    // Reduces 8 rows to max in LREG0/LREG4, optionally stores result.
    auto reduce_8_rows = [instr_mod_index](const std::uint32_t val_base, const std::uint32_t idx_base, const bool store_result) __attribute__((always_inline))
    {
        // data - precomputed base address eliminates repeated arithmetic
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base + 0);  // Row 0 and 1
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base + 4);  // Row 2 and 3
        TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base + 8);  // Row 4 and 5
        TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base + 12); // Row 6 and 7
        // index
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, idx_base + 0);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, idx_base + 4);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, idx_base + 8);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, idx_base + 12);

        // Reduce 8 rows to max in LREG0/LREG4 via replay buffer
        lltt::replay(0, 7);

        // Only store when necessary - caller controls this
        if (store_result)
        {
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base + 0);
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, idx_base + 0);
        }
        // Result: Max of 8 rows in LREG0 (values) and LREG4 (indices)
    };

    // OPTIMIZATION: Flattened process_16_rows - eliminates nested lambda overhead
    // and removes redundant store-load pairs.
    // Note: After reducing the second 8-row block, the result is already in LREG0/LREG4.
    // We only need to reload the first block's result into LREG1/LREG5 for the final swap.
    // store_result: if false, result stays in LREG0/LREG4 for caller to use directly.
    auto process_16_rows = [&reduce_8_rows, values_tile_offset, indices_tile_offset, eight_row_offset, instr_mod_index](
                               const std::uint32_t base_offset, const std::uint32_t col_offset, const bool store_result) __attribute__((always_inline))
    {
        // Precompute base addresses for both 8-row blocks
        const std::uint32_t val_base_first  = values_tile_offset + base_offset + col_offset;
        const std::uint32_t idx_base_first  = indices_tile_offset + base_offset + col_offset;
        const std::uint32_t val_base_second = values_tile_offset + eight_row_offset + base_offset + col_offset;
        const std::uint32_t idx_base_second = indices_tile_offset + eight_row_offset + base_offset + col_offset;

        // First 8 rows: reduce and STORE (we need to free registers for second block)
        reduce_8_rows(val_base_first, idx_base_first, true);

        // Second 8 rows: reduce but DON'T STORE - keep result in LREG0/LREG4
        reduce_8_rows(val_base_second, idx_base_second, false);

        // Now: LREG0/LREG4 contains Max(R8-15), need to load Max(R0-7) into LREG1/LREG5
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base_first);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, idx_base_first);

        // Swap to get Max(R0-15) in LREG0/LREG4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Only store if needed - caller may use LREG0/LREG4 directly
        if (store_result)
        {
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, idx_base_first);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_base_first);
        }
    };

    // Final swap: combine first 16 rows with second 16 rows.
    // OPTIMIZATION: After process_16_rows(sixteen_row_offset, col), Max(R16-31) is already in LREG0/LREG4.
    // We only need to load Max(R0-15) into LREG1/LREG5, saving 2 loads per column.
    auto final_swap = [values_tile_offset, indices_tile_offset, values_accum_tile_offset, indices_accum_tile_offset, instr_mod_index, chunk](
                          const std::uint32_t col_offset) __attribute__((always_inline))
    {
        // Precompute addresses
        const std::uint32_t val_first = values_tile_offset + col_offset;
        const std::uint32_t idx_first = indices_tile_offset + col_offset;

        // LREG0/LREG4 already contains Max(R16-31) from the previous process_16_rows call
        // Only need to load Max(R0-15) into LREG1/LREG5
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_first); // Max(R0-15) for F0,1
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, idx_first);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // LREG0 contains Max(R0-31) for F0,1

        if constexpr (accumulate)
        {
            const std::uint32_t val_accum = values_accum_tile_offset + col_offset;
            const std::uint32_t idx_accum = indices_accum_tile_offset + col_offset;
            if (chunk > 0)
            { // for all but the first chunk we need to load the previous result from DST 1 and 3 and do a max with the current result in DST 0 and 2
                TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_accum); // previous accumulated value
                TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, idx_accum);            // previous accumulated index
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);        // LREG0 contains max of current and previous value
            }
            // for each chunk we store the running result to DST 1 and 3
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, idx_accum);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_accum);
        }

        // store the final result to DST 0 (data) and DST 2 (indices)
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, idx_first);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, val_first);
    };

    // OPTIMIZATION: Process each column completely before moving to the next.
    // This allows the second process_16_rows to leave Max(R16-31) in LREG0/LREG4,
    // which final_swap can use directly without reloading.
    // Saves 2 stores + 2 loads per column = 4 stores + 4 loads total.
    constexpr int even_column_offset = 0;
    constexpr int odd_column_offset  = 2;

    // Even columns: process rows 0-15, then 16-31, then final swap
    process_16_rows(0, even_column_offset, true);                   // Store Max(R0-15) for final_swap to load
    process_16_rows(sixteen_row_offset, even_column_offset, false); // Keep Max(R16-31) in LREG0/LREG4
    final_swap(even_column_offset);                                 // Uses LREG0/LREG4 directly

    // Odd columns: process rows 0-15, then 16-31, then final swap
    process_16_rows(0, odd_column_offset, true);                   // Store Max(R0-15) for final_swap to load
    process_16_rows(sixteen_row_offset, odd_column_offset, false); // Keep Max(R16-31) in LREG0/LREG4
    final_swap(odd_column_offset);                                 // Uses LREG0/LREG4 directly
}

template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void _init_max_pool_with_indices_()
{
    // Set bit [2] of the SFPU_CONTROL_REG to enable Destination Index Tracking Mode:
    // LREGs 4-7 will be treated as indices corresponding to the values in LREGs 0-3,
    // and LREGs 4-7 will mirror the movement of the values in LREGs 0-3;
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // Program replay buffer for row major layout
        load_replay_buf(
            0,
            7,
            []
            {
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

                TTI_SFPTRANSP(0, 0, 0, 0);

                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                TTI_SFPTRANSP(0, 0, 0, 0);
            });
    }
    else
    {
        // Program replay buffer for tiled layout (original)
        load_replay_buf(
            0,
            7,
            []
            {
                // Values have been loaded such that 4 rows of Dest occupy the 4 lanes of each LREG
                // To sort those 4 rows, we transpose the SFPU LREGs  to put elements of 4 rows of each column into separate LREGs of each unit of SFPU
                TTI_SFPTRANSP(0, 0, 0, 0);

                // Sort the 4 rows of Dest, placing the max value into LREG0 (index into LREG4)
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

                // Transpose the LREGs back
                TTI_SFPTRANSP(0, 0, 0, 0);

                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
            });
    }
}

} // namespace sfpu
} // namespace ckernel
