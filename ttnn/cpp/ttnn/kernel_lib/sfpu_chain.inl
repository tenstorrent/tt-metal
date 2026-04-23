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
ALWI void Load<CB, Slot, Policy, Reconfig>::exec(uint32_t offset) const {
    if constexpr (do_wait) {
        cb_wait_front(CB, cb_tile_idx_ + 1);
    }
    copy_tile(CB, cb_tile_idx_, static_cast<uint32_t>(Slot) + offset);
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

    // Packer reconfiguration once before the loop.
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Hoist Load::init once before the tile loop when safe. Otherwise Load::apply
    // (init+exec) fires per chunk inside the cascade — required when the chain
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

    // Pack amortization: run as many chain iterations as fit in DEST inside
    // one tile_regs_acquire/commit/wait/release window, then pack as a burst.
    //
    // Stride — how much each chain iteration's slot block shifts in DEST:
    //   Non-optimized (current) mode: stride = pack_slot + 1 (= 1 with the
    //     Dst::D0 convention). The result slot is spaced per-iter; scratch
    //     slots from iter k naturally get overwritten by iter k+1's loads
    //     (iter k's scratch is dead once iter k commits its result).
    //     Compact: K outputs + scratch overhead live simultaneously.
    //   Optimized mode (future, init-once/exec-K): stride = max_dst + 1 —
    //     all K iterations' scratch slots live concurrently because each op
    //     fires K times before the next op begins.
    //
    // Budget: max DEST slot touched across the chunk is
    //   (K - 1) * stride + max_dst. Must stay below DEST_AUTO_LIMIT.
    // → K = (DEST_AUTO_LIMIT - max_dst + pack_slot) / stride + 1.
    //   With pack_slot = 0, stride = 1: K = DEST_AUTO_LIMIT - max_dst.
    //
    // Every op's init/exec still fires per iteration — behaviourally the
    // same as the old per-tile pipeline, just with more tiles sharing one
    // DEST acquire. The door is open for init-once/exec-K optimization by
    // swapping this inner loop for a batched variant.
    ASSERT(static_cast<uint32_t>(pack_slot) == 0);
    constexpr uint32_t pack_base = 0u;
    constexpr uint32_t stride = pack_base + 1u;  // non-optimized mode
    constexpr uint32_t max_dst = Chain::max_dst;
    constexpr uint32_t iter_per_chunk = (DEST_AUTO_LIMIT > max_dst) ? (DEST_AUTO_LIMIT - max_dst) : 1u;
    static_assert(max_dst < DEST_AUTO_LIMIT, "chain max_dst exceeds DEST capacity");

    uint32_t tiles_processed = 0;
    for (uint32_t base = 0; base < num_tiles; base += iter_per_chunk) {
        const uint32_t this_chunk = (base + iter_per_chunk <= num_tiles) ? iter_per_chunk : (num_tiles - base);

        tile_regs_acquire();
        for (uint32_t k = 0; k < this_chunk; ++k) {
            const uint32_t dst_offset = k * stride;
            if constexpr (hoist_load_init) {
                chain.apply_no_load_init(dst_offset);
            } else {
                chain.apply(dst_offset);
            }
        }
        tile_regs_commit();
        tile_regs_wait();

        if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
            for (uint32_t k = 0; k < this_chunk; ++k) {
                cb_reserve_back(ocb, 1);
                pack_tile(pack_base + k * stride, ocb);
                cb_push_back(ocb, 1);
            }
        } else {
            // Bulk: pack to absolute output tile index. cb_reserve_back already
            // covered all num_tiles; cb_push_back issued once after the loop.
            for (uint32_t k = 0; k < this_chunk; ++k) {
                pack_tile(pack_base + k * stride, ocb, tiles_processed + k);
            }
        }

        tile_regs_release();
        tiles_processed += this_chunk;
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
