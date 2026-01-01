// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Configure the advancing output store for supplied-index packing.
 * @note Call after shared SFPU initialization and operand copy initialization, before
 *       @ref _topk_xl_pack_supplied_indices_. Programs ADDR_MOD_0; leaves the TopK
 *       bitonic address modifiers intact. Reinitialize datacopy before the next copy.
 */
inline void _topk_xl_pack_supplied_indices_init_()
{
    // Keep ADDR_MOD_6's +32 stride live for the subsequent TopK sort/rebuild.
    // Using it for +2 here would require restoring it in addition to the MOP.
    // ADDR_MOD_0 is unused by those bitonic kernels; datacopy is reinitialized separately.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

/**
 * @brief Fuse BF16-expanded scores with supplied low-16-bit indices in FP32 DEST.
 * @tparam K: Number of entries, 512, 1024, or 2048.
 * @param scores: First score tile in DEST.
 * @param indices: First raw UInt32 index tile in DEST.
 * @param output: First packed output tile in DEST.
 * @param score_half: For K=512, select the first or second 512-element score half; IDs always start at indices[0].
 * @note Use the shared SFPU params wrapper with VectorMode::RC_custom and initialize
 *       the SFPU, then call @ref _topk_xl_pack_supplied_indices_init_ before calling.
 *       Load indices without numeric conversion. Output may
 *       alias either input at the same start; other overlapping regions are unsupported.
 *       UINT32_MAX maps to negative-infinity padding. K=512 also clears the output tail.
 */
template <std::uint32_t K>
inline void _topk_xl_pack_supplied_indices_(const std::uint32_t scores, const std::uint32_t indices, const std::uint32_t output, const std::uint32_t score_half)
{
    static_assert(K == 512 || K == 1024 || K == 2048);
    // Each SFPI slot addresses 32 elements (two physical DEST rows).
    constexpr std::uint32_t elements_per_slot = 32;
    constexpr std::uint32_t tile_slots        = 32;
    constexpr auto sentinel_id                = -1;
    constexpr std::uint32_t half_slots        = tile_slots / 2;
    const std::uint32_t score_offset          = scores * tile_slots + (K == 512 ? score_half * half_slots : 0);
    const std::uint32_t index_offset          = indices * tile_slots;
    const std::uint32_t output_offset         = output * tile_slots;
    // Load the masks and padding once instead of per slot (four SFPLOADIs per slot).
    // The sentinel needs no register: the compare folds it into an immediate add.
    const sfpi::vInt score_mask(0xffff0000u);
    const sfpi::vInt index_mask(0xffffu);
    const sfpi::vInt padding(static_cast<std::int32_t>(0xff80ffffu));
#pragma GCC unroll 8
    for (std::uint32_t i = 0; i < K / elements_per_slot; ++i)
    {
        sfpi::vInt score  = sfpi::dst_reg[score_offset].mode<sfpi::DataLayout::I32>();
        sfpi::vInt id     = sfpi::dst_reg[index_offset].mode<sfpi::DataLayout::I32>();
        sfpi::vInt packed = (score & score_mask) | (id & index_mask);
        v_if (id == sentinel_id)
        {
            packed = padding;
        }
        v_endif;
        sfpi::dst_reg[output_offset].mode<sfpi::DataLayout::I32>(ADDR_MOD_0) = packed;
    }
    if constexpr (K == 512)
    {
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < half_slots; ++i)
        {
            sfpi::dst_reg[output_offset].mode<sfpi::DataLayout::I32>(ADDR_MOD_0) = padding;
        }
    }
}

} // namespace ckernel::sfpu
