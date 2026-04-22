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
    // Program the copy_tile MOP for this CB, and (optionally) the SRCA data format.
    // The pipeline hoists this call out of the tile loop when the chain is
    // hoist-safe (see chain_is_hoist_safe_v). Otherwise apply() fires it per tile.
    copy_tile_to_dst_init_short(CB);
    if constexpr (do_reconfig) {
        reconfig_data_format_srca(CB);
    }
}

template <uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig>
ALWI void Load<CB, Slot, Policy, Reconfig>::exec() const {
    if constexpr (do_wait) {
        cb_wait_front(CB, cb_tile_idx_ + 1);
    }
    copy_tile(CB, cb_tile_idx_, static_cast<uint32_t>(Slot));
    if constexpr (do_pop) {
        cb_pop_front(CB, 1);
    }
    // WaitUpfrontPopAtEnd: pipeline bulk-waits N tiles before the loop and
    // bulk-pops N after via wait_upfront/pop_upfront. Here we only advance the
    // rising index into the pre-waited block.
    if constexpr (is_upfront) {
        cb_tile_idx_++;
    }
}

// =============================================================================
// Internal Pipeline Helpers
// =============================================================================

namespace detail {

constexpr bool sfpu_reconfig_output(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::OUTPUT;
}

}  // namespace detail

// =============================================================================
// Pipeline Implementation
// =============================================================================

template <
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Chain>
ALWI void sfpu_pipeline(
    Chain& chain,
    uint32_t ocb,
    uint32_t num_tiles,
    Dst pack_slot) {
    static_assert(
        !chain_has_duplicate_upfront_cbs_v<Chain>,
        "sfpu_pipeline: chain contains two or more WaitUpfrontPopAtEnd Loads on the same CB; "
        "the pipeline would double-pop. Use a single upfront Load per CB, or duplicate via "
        "WaitNoPop/NoWaitPop fan-out.");
    ASSERT(num_tiles > 0);
    ASSERT(static_cast<uint32_t>(pack_slot) < Chain::stride);

    // Packer reconfiguration once before the loop.
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Hoist Load::init once before the tile loop when safe. Otherwise Load::apply
    // (init+exec) fires per tile inside the cascade — required when the chain
    // clobbers copy_tile state (e.g. DestReuseOp) or Loads use multiple CBs.
    constexpr bool hoist_load_init = chain_is_hoist_safe_v<Chain>;
    if constexpr (hoist_load_init) {
        chain.init_any_load();
    }

    // Bulk wait for every WaitUpfrontPopAtEnd Load's CB (no-op for other policies).
    chain.wait_upfront(num_tiles);

    // Bulk output: reserve all tiles upfront.
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    // One tile per acquire cycle — DEST slots are fixed at compile time.
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        if constexpr (hoist_load_init) {
            chain.apply_no_load_init();
        } else {
            chain.apply();
        }
        tile_regs_commit();
        tile_regs_wait();

        if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
            cb_reserve_back(ocb, 1);
        }
        pack_tile(static_cast<uint32_t>(pack_slot), ocb);
        if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
            cb_push_back(ocb, 1);
        }

        tile_regs_release();
    }

    // Bulk output: push all tiles at end.
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_push_back(ocb, num_tiles);
    }

    // Bulk pop every WaitUpfrontPopAtEnd Load's CB (no-op for other policies).
    chain.pop_upfront(num_tiles);

    // Reset WaitUpfrontPopAtEnd Load indices so the chain can be reused across blocks.
    chain.reset_tile_idx();
}

// =============================================================================
// sfpu_op Implementation
// =============================================================================

template <
    uint32_t ICB,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op) {
    auto chain = sfpu_chain(Load<ICB, Dst::D0>{}, op);
    sfpu_pipeline<output_policy, reconfig>(chain, ocb, num_tiles);
}

}  // namespace compute_kernel_lib
