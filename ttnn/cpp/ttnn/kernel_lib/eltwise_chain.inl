// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.inl
 * @brief Method definitions for eltwise_chain.hpp.
 *
 * Only included by eltwise_chain.hpp.
 */

namespace compute_kernel_lib::eltwise {

namespace detail {

constexpr bool reconfig_input(EltwiseDataFormatReconfig m) {
    return m == EltwiseDataFormatReconfig::INPUT ||
           m == EltwiseDataFormatReconfig::INPUT_AND_OUTPUT;
}
constexpr bool reconfig_output(EltwiseDataFormatReconfig m) {
    return m == EltwiseDataFormatReconfig::OUTPUT ||
           m == EltwiseDataFormatReconfig::INPUT_AND_OUTPUT;
}

/// Pipeline-internal hook that gates calls to CopyTile's mutable bookkeeping.
/// Lives outside the CopyTile struct so the friendship is narrow.
struct EltwisePipelineDetail {
    template <typename Element>
    ALWI static void advance(const Element& e) {
        if constexpr (is_copy_tile_op_v<Element>) {
            e._pipeline_advance();
        }
    }
    template <typename Element>
    ALWI static void reset(const Element& e) {
        if constexpr (is_copy_tile_op_v<Element>) {
            e._pipeline_reset();
        }
    }
};

}  // namespace detail

// =============================================================================
// CopyTile method definitions
// =============================================================================

template <uint32_t CB, Dst Slot, CopyTilePolicy Policy, CopyTileReconfig Reconfig>
ALWI void CopyTile<CB, Slot, Policy, Reconfig>::init() const {
    if constexpr (Reconfig == CopyTileReconfig::Input) {
        reconfig_data_format_srca(CB);
    }
    copy_tile_to_dst_init_short(CB);
}

template <uint32_t CB, Dst Slot, CopyTilePolicy Policy, CopyTileReconfig Reconfig>
ALWI void CopyTile<CB, Slot, Policy, Reconfig>::exec(uint32_t offset) const {
    constexpr bool waits_per_tile =
        (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop);
    constexpr bool pops_per_tile =
        (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop);

    if constexpr (Policy == CopyTilePolicy::CumulativeWait) {
        // wait for prior + this tile to be visible.
        cb_wait_front(CB, cb_tile_idx_ + 1);
        copy_tile(CB, cb_tile_idx_, dst_idx + offset);
        // pop: never (caller pops bulk at end).
    } else if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
        // upfront wait fired once outside the loop. Index is whatever the
        // pipeline has counted so far.
        copy_tile(CB, cb_tile_idx_, dst_idx + offset);
        // pop: deferred to upfront-pop pass.
    } else {
        if constexpr (waits_per_tile) {
            cb_wait_front(CB, 1);
        }
        copy_tile(CB, 0, dst_idx + offset);
        if constexpr (pops_per_tile) {
            cb_pop_front(CB, 1);
        }
    }
}

template <uint32_t CB, Dst Slot, CopyTilePolicy Policy, CopyTileReconfig Reconfig>
ALWI void CopyTile<CB, Slot, Policy, Reconfig>::_pipeline_advance() const {
    if constexpr (Policy == CopyTilePolicy::CumulativeWait ||
                  Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
        cb_tile_idx_++;
    }
}

template <uint32_t CB, Dst Slot, CopyTilePolicy Policy, CopyTileReconfig Reconfig>
ALWI void CopyTile<CB, Slot, Policy, Reconfig>::_pipeline_reset() const {
    cb_tile_idx_ = 0;
}

// =============================================================================
// FillScalar / FillConst / CopyDest
// =============================================================================

template <Dst Slot>
ALWI void FillScalar<Slot>::init() const { fill_tile_init(); }
template <Dst Slot>
ALWI void FillScalar<Slot>::call(uint32_t d0) const { fill_tile(d0, value); }

template <uint32_t Bits, Dst Slot>
ALWI void FillConst<Bits, Slot>::init() const { fill_tile_init(); }
template <uint32_t Bits, Dst Slot>
ALWI void FillConst<Bits, Slot>::call(uint32_t d0) const { fill_tile_bitcast(d0, Bits); }

template <Dst Src, Dst Dest, DataFormat DF>
ALWI void CopyDest<Src, Dest, DF>::init() const { copy_dest_values_init(); }
template <Dst Src, Dst Dest, DataFormat DF>
ALWI void CopyDest<Src, Dest, DF>::exec(uint32_t offset) const {
    copy_dest_values<DF>(in0 + offset, out + offset);
}

// =============================================================================
// EltwiseChain method definitions — recurse over the type list
// =============================================================================

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::apply(uint32_t offset) const {
    first.apply(offset);
    rest.apply(offset);
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::apply_init_only() const {
    first.init();
    rest.apply_init_only();
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::apply_exec_only(uint32_t offset) const {
    first.exec(offset);
    rest.apply_exec_only(offset);
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::wait_upfront(uint32_t num_tiles) const {
    if constexpr (First::is_upfront) {
        cb_wait_front(First::cb, num_tiles);
    }
    rest.wait_upfront(num_tiles);
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::pop_upfront(uint32_t num_tiles) const {
    if constexpr (First::is_upfront) {
        cb_pop_front(First::cb, num_tiles);
    }
    rest.pop_upfront(num_tiles);
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::advance_cumulative() const {
    detail::EltwisePipelineDetail::advance(first);
    rest.advance_cumulative();
}

template <typename First, typename... Rest>
ALWI void EltwiseChain<First, Rest...>::reset_cumulative_and_upfront() const {
    detail::EltwisePipelineDetail::reset(first);
    rest.reset_cumulative_and_upfront();
}

// =============================================================================
// eltwise_pipeline implementation
// =============================================================================

template <
    EltwiseOutputPolicy OutPolicy,
    EltwiseDataFormatReconfig Reconfig,
    typename Chain>
ALWI void eltwise_pipeline(
    Chain chain, uint32_t ocb, uint32_t num_tiles, Dst pack_slot) {
    ASSERT(num_tiles > 0);

    constexpr uint32_t chain_stride = Chain::stride;
    static_assert(chain_stride <= DST_HW_CEILING,
                  "chain stride exceeds DEST hw ceiling (16)");
    ASSERT(chain_stride <= DEST_AUTO_LIMIT);
    ASSERT(static_cast<uint32_t>(pack_slot) < chain_stride);

    constexpr uint32_t first_cb = detail::FirstCopyTileCB<Chain>::value;

    // 1. Reconfig.
    if constexpr (detail::reconfig_input(Reconfig) && chain_has_any_copy_tile_v<Chain>) {
        reconfig_data_format_srca(first_cb);
    }
    if constexpr (detail::reconfig_output(Reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // 2. Initial unpacker init for the first copy_tile CB.
    if constexpr (chain_has_any_copy_tile_v<Chain>) {
        copy_tile_to_dst_init_short(first_cb);
    }

    // 3. Upfront waits — for every is_upfront element.
    chain.wait_upfront(num_tiles);

    // Bulk output reserve.
    if constexpr (OutPolicy == EltwiseOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    const uint32_t pack_idx = static_cast<uint32_t>(pack_slot);

    // 4. Per-tile loop.
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        chain.apply(0);
        tile_regs_commit();
        tile_regs_wait();

        if constexpr (OutPolicy == EltwiseOutputPolicy::PerTile) {
            cb_reserve_back(ocb, 1);
            pack_tile(pack_idx, ocb);
            cb_push_back(ocb, 1);
        } else {  // Bulk
            pack_tile(pack_idx, ocb, i);
        }

        tile_regs_release();
        chain.advance_cumulative();
    }

    if constexpr (OutPolicy == EltwiseOutputPolicy::Bulk) {
        cb_push_back(ocb, num_tiles);
    }

    // 5. Upfront pops.
    chain.pop_upfront(num_tiles);

    // 6. Reset CopyTile bookkeeping so the chain instance is reusable.
    chain.reset_cumulative_and_upfront();
}

}  // namespace compute_kernel_lib::eltwise
