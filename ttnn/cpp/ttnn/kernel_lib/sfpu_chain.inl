// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file sfpu_chain.inl
 * @brief Out-of-line implementations for Load, sfpu_pipeline, and sfpu_op.
 * Should only be included by sfpu_chain.hpp.
 */

namespace compute_kernel_lib {

using namespace ckernel;

// =============================================================================
// Load Implementation
// =============================================================================

template <uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig>
ALWI void Load<CB, Slot, Policy, Reconfig>::init() const {
    // No-op: copy_tile_to_dst_init is handled once by the pipeline before the tile loop.
    // This keeps init() uniform with compute ops but avoids redundant re-initialization.
}

template <uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig>
ALWI void Load<CB, Slot, Policy, Reconfig>::exec(uint32_t offset) const {
    if constexpr (do_wait) {
        // Wait for enough tiles to cover cb_tile_idx (minimum 1 tile for index 0).
        cb_wait_front(CB, cb_tile_idx + 1);
    }
    if constexpr (Reconfig == LoadReconfig::Input) {
        // Load always reads through unpack A (copy_tile).
        reconfig_data_format_srca(CB);
    }
    copy_tile(CB, cb_tile_idx, static_cast<uint32_t>(Slot) + offset);
    if constexpr (do_pop) {
        cb_pop_front(CB, 1);
    }
}

// =============================================================================
// Internal Pipeline Helpers
// =============================================================================

namespace detail {

constexpr bool sfpu_reconfig_input(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::INPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

constexpr bool sfpu_reconfig_output(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::OUTPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

/** @brief Get the CB of the first Load in a chain (for input reconfig) */
template <typename Chain>
struct FirstLoadCB {
    static constexpr uint32_t value = 0;
};
// Non-load first element: recurse
template <typename First, typename... Rest>
struct FirstLoadCB<SfpuChain<First, Rest...>> {
    static constexpr uint32_t value = FirstLoadCB<SfpuChain<Rest...>>::value;
};
// Load first element: found it
template <uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig, typename... Rest>
struct FirstLoadCB<SfpuChain<Load<CB, Slot, Policy, Reconfig>, Rest...>> {
    static constexpr uint32_t value = CB;
};

}  // namespace detail

// =============================================================================
// Pipeline Implementation
// =============================================================================

template <
    SfpuBatching batching,
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Chain>
ALWI void sfpu_pipeline(
    Chain chain,
    uint32_t ocb,
    uint32_t num_tiles,
    Dst pack_slot) {
    ASSERT(num_tiles > 0);

    constexpr uint32_t chain_stride = Chain::stride;
    constexpr uint32_t batch_size = (batching == SfpuBatching::Disabled)
        ? 1 : (DEST_AUTO_LIMIT / chain_stride);
    static_assert(batch_size >= 1, "chain stride exceeds DEST capacity");

    ASSERT(static_cast<uint32_t>(pack_slot) < chain_stride);

    // Data format reconfiguration (once before the tile loop)
    if constexpr (detail::sfpu_reconfig_input(reconfig)) {
        reconfig_data_format_srca(detail::FirstLoadCB<Chain>::value);
    }
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Initialize unpacker for the first Load's CB (once, before the tile loop)
    copy_tile_to_dst_init_short(detail::FirstLoadCB<Chain>::value);

    // Bulk output: reserve all tiles upfront
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    // Tile loop: full init+exec (apply) every tile. Multi-op chains require per-op
    // re-init anyway (SFPU inits interfere), and the single-op case is rare enough
    // that amortising it across tiles via exec_only isn't worth the extra complexity.
    // Batched path packs multiple tiles per acquire/release cycle.
    for (uint32_t i = 0; i < num_tiles; i += batch_size) {
        const uint32_t actual =
            (batch_size == 1) ? 1 : (((i + batch_size) <= num_tiles) ? batch_size : (num_tiles - i));

        tile_regs_acquire();

        for (uint32_t k = 0; k < actual; ++k) {
            chain.apply(k * chain_stride);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t k = 0; k < actual; ++k) {
            if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
                cb_reserve_back(ocb, 1);
            }
            pack_tile(static_cast<uint32_t>(pack_slot) + k * chain_stride, ocb);
            if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
                cb_push_back(ocb, 1);
            }
        }

        tile_regs_release();
    }

    // Bulk output: push all tiles at end
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_push_back(ocb, num_tiles);
    }
}

// =============================================================================
// sfpu_op Implementation
// =============================================================================

template <
    uint32_t ICB,
    SfpuBatching batching,
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op) {
    auto chain = sfpu_chain(Load<ICB, Dst::D0>{}, op);
    sfpu_pipeline<batching, input_policy, output_policy, reconfig>(chain, ocb, num_tiles);
}

}  // namespace compute_kernel_lib
