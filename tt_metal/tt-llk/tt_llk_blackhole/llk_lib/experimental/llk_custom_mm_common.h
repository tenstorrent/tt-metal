// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

/**
 * Where custom_mm puts its [{1, 2, 4, 8}, 32] output tiles in DEST. Each tile has two output faces of up to 8 rows.
 *
 *   Default          tiles 64 rows apart, output face 1 at row 16
 *   DefaultSplitAcc  as Default, accumulating the second k-face's partials at rows 8 and 24 for finalize to merge
 *   Dense            tiles 32 rows apart, output face 1 at row 16
 *   DenseSplitAcc    as Dense, with the split partials at rows 8 and 24
 *   Compact          tiles 16 rows apart, output face 1 at row 8, no room for split partials
 *
 * Init and execute must use the same layout, on the math and pack threads alike.
 *
 * The tile spacing sets how many output tiles, and so how wide a ct_dim, fit in DEST: 8 Default, 16 Dense or 32
 * Compact tiles in a 16-bit SyncHalf DEST (512 rows), twice that with SyncFull and half that with a 32-bit DEST.
 * See custom_mm_fits_dest.
 */
enum class CustomMmDestLayout : std::uint8_t
{
    Default,
    DefaultSplitAcc,
    Dense,
    DenseSplitAcc,
    Compact,
};

// DEST rows from one output tile to the next; dst_index counts tiles of this many rows.
constexpr std::uint32_t custom_mm_tile_rows(const CustomMmDestLayout layout)
{
    switch (layout)
    {
        case CustomMmDestLayout::Dense:
        case CustomMmDestLayout::DenseSplitAcc:
            return 32;
        case CustomMmDestLayout::Compact:
            return 16;
        default:
            return 64;
    }
}

constexpr bool custom_mm_split_acc(const CustomMmDestLayout layout)
{
    return layout == CustomMmDestLayout::DefaultSplitAcc || layout == CustomMmDestLayout::DenseSplitAcc;
}

// Widest ct_dim the unpack MOP covers: the full unpack and 8 reuses, two more runs of 8 reuses, then 9 reuses ending
// in the final one. DEST capacity usually bounds it first.
constexpr std::uint32_t CUSTOM_MM_MAX_CT_DIM = 34;
// Widest kt_dim the SrcB address counters cover: 512 face increments split over the 8-bit Y and Z counters.
constexpr std::uint32_t CUSTOM_MM_MAX_KT_DIM = 256;

} // namespace ckernel
