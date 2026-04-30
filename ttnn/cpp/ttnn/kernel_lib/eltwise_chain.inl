// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.inl
 * @brief Implementation of eltwise_pipeline.
 *
 * Composes directly from compute_kernel_api primitives (cb_wait_front /
 * cb_pop_front / cb_reserve_back / cb_push_back, tile_regs_*, copy_tile,
 * pack_tile, copy_tile_to_dst_init_short). NO calls into legacy
 * sfpu_helpers.{hpp,inl} or binary_op_helpers.{hpp,inl}.
 */

#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/assert.h"

namespace compute_kernel_lib {

namespace detail {

// =============================================================================
// Same-CB dedup helpers — for a CopyTile element at index Idx in tuple T,
// determine whether it is the first / last element with `cb == CopyTile::cb`
// among all CopyTile elements in the tuple.
// =============================================================================

template <std::size_t Idx, typename Tuple>
constexpr bool is_first_user_of_cb() {
    using This = std::tuple_element_t<Idx, Tuple>;
    if constexpr (!is_copy_tile_op_v<This>) {
        return false;
    } else {
        bool first = true;
        constexpr std::size_t N = std::tuple_size_v<Tuple>;
        // Check all elements before this one. If any earlier CopyTile shares
        // the same CB, this is not the first.
        [&]<std::size_t... Js>(std::index_sequence<Js...>) {
            ((Js < Idx ? (
                  is_copy_tile_op_v<std::tuple_element_t<Js, Tuple>> &&
                          std::tuple_element_t<Js, Tuple>::cb == This::cb
                      ? (first = false)
                      : false)
                       : false),
             ...);
        }(std::make_index_sequence<N>{});
        return first;
    }
}

template <std::size_t Idx, typename Tuple>
constexpr bool is_last_user_of_cb() {
    using This = std::tuple_element_t<Idx, Tuple>;
    if constexpr (!is_copy_tile_op_v<This>) {
        return false;
    } else {
        bool last = true;
        constexpr std::size_t N = std::tuple_size_v<Tuple>;
        [&]<std::size_t... Js>(std::index_sequence<Js...>) {
            ((Js > Idx ? (
                  is_copy_tile_op_v<std::tuple_element_t<Js, Tuple>> &&
                          std::tuple_element_t<Js, Tuple>::cb == This::cb
                      ? (last = false)
                      : false)
                       : false),
             ...);
        }(std::make_index_sequence<N>{});
        return last;
    }
}

// =============================================================================
// Per-element op handlers
// =============================================================================

// Resolve the CB tile index a CopyTile element should read this iteration.
template <typename CT>
ALWI uint32_t resolve_cb_tile_idx(const CT& ct, uint32_t loop_i) {
    if constexpr (CT::index_mode == CbIndexMode::FirstTile) {
        // For single-tile-window policies tile 0 is always at the head. For
        // upfront policies "tile 0" is the first tile of the upfront window —
        // ASSERT-protected (caller responsibility).
        (void)ct;
        (void)loop_i;
        return 0u;
    } else if constexpr (CT::index_mode == CbIndexMode::BlockIter) {
        (void)ct;
        return loop_i;
    } else if constexpr (CT::index_mode == CbIndexMode::Pinned) {
        (void)loop_i;
        return ct.cb_tile_idx_runtime;
    } else {  // Absolute
        (void)loop_i;
        return ct.cb_tile_idx_runtime;
    }
}

// Run a CopyTile element's init (copy_tile_to_dst_init_short + optional reconfig).
template <typename CT>
ALWI void copy_tile_init(const CT&) {
    if constexpr (CT::reconfig == CopyTileReconfig::Input) {
        reconfig_data_format_srca(CT::cb);
    }
    copy_tile_to_dst_init_short(CT::cb);
}

// Per-tile body for a CopyTile element. `is_first` and `is_last` come from the
// dedup check; the wait fires only on the first user, pop only on the last.
template <typename CT, bool RunInit>
ALWI void run_copy_tile(const CT& ct, uint32_t loop_i, bool is_first, bool is_last) {
    if constexpr (RunInit) {
        copy_tile_init(ct);
    }

    // Per-tile wait (only if policy says so, and only for the first user).
    if constexpr (CT::policy_does_per_tile_wait) {
        if (is_first) {
            cb_wait_front(CT::cb, 1);
        }
    }

    const uint32_t in_idx = resolve_cb_tile_idx(ct, loop_i);
    copy_tile(CT::cb, in_idx, CT::dst_idx);

    // Per-tile pop (only if policy says so, and only for the last user).
    if constexpr (CT::policy_does_per_tile_pop) {
        if (is_last) {
            cb_pop_front(CT::cb, 1);
        }
    }
}

// Upfront-wait emitted once before the loop. For CumulativeWaitUpfrontEndPop
// this is replaced by per-iter growing waits below.
template <typename CT>
ALWI void run_upfront_wait(const CT&, uint32_t num_tiles) {
    if constexpr (CT::policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
        cb_wait_front(CT::cb, num_tiles);
    }
}

// Cumulative per-iteration growing wait. Emitted INSIDE the loop, before the
// CopyTile runs.
template <typename CT>
ALWI void run_cumulative_wait(const CT&, uint32_t loop_i) {
    if constexpr (CT::policy == CopyTilePolicy::CumulativeWaitUpfrontEndPop) {
        cb_wait_front(CT::cb, loop_i + 1);
    }
}

// Upfront pop emitted once after the loop.
template <typename CT>
ALWI void run_upfront_pop(const CT&, uint32_t num_tiles) {
    if constexpr (CT::policy_does_upfront_pop) {
        cb_pop_front(CT::cb, num_tiles);
    }
}

// Reset internal counters at end-of-block.
template <typename CT>
ALWI void reset_runtime_state(const CT& ct) {
    ct.cb_tile_idx_ = 0;
}

// =============================================================================
// Tuple iteration helpers
// =============================================================================

template <typename Tuple, typename Fn, std::size_t... Is>
ALWI void for_each_indexed_impl(Tuple& t, Fn&& fn, std::index_sequence<Is...>) {
    (fn.template operator()<Is>(std::get<Is>(t)), ...);
}

template <typename Tuple, typename Fn>
ALWI void for_each_indexed(Tuple& t, Fn&& fn) {
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_indexed_impl(t, std::forward<Fn>(fn), std::make_index_sequence<N>{});
}

}  // namespace detail

// =============================================================================
// eltwise_pipeline implementation
// =============================================================================

template <uint32_t OutCB, bool EnableHoist, typename Chain>
ALWI void eltwise_pipeline(uint32_t num_tiles, Chain chain) {
    using ChainT = Chain;
    using TupleT = decltype(chain.elements);
    constexpr std::size_t N = std::tuple_size_v<TupleT>;

    // Hoisting is allowed only when explicitly opted in AND the chain is
    // structurally safe. Otherwise inits run per-tile.
    constexpr bool do_hoist = EnableHoist && chain_is_hoist_safe_v<ChainT>;

    // -----------------------------------------------------------------
    // Phase 1: emit upfront cb_wait_front for any WaitUpfrontPopAtEnd elements.
    // -----------------------------------------------------------------
    detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
        using E = std::remove_reference_t<decltype(elem)>;
        if constexpr (detail::is_copy_tile_op_v<E>) {
            // Only the first user emits the wait — same-CB dedup applies.
            constexpr bool is_first = detail::is_first_user_of_cb<Idx, TupleT>();
            if constexpr (is_first) {
                detail::run_upfront_wait(elem, num_tiles);
            }
        }
    });

    // -----------------------------------------------------------------
    // Phase 2: hoisted inits (when do_hoist).
    // -----------------------------------------------------------------
    if constexpr (do_hoist) {
        // First emit any CopyTile inits (paired with their CB).
        detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
            using E = std::remove_reference_t<decltype(elem)>;
            if constexpr (detail::is_copy_tile_op_v<E>) {
                constexpr bool is_first = detail::is_first_user_of_cb<Idx, TupleT>();
                if constexpr (is_first) {
                    detail::copy_tile_init(elem);
                }
            }
        });
        // Then non-CopyTile op inits.
        detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
            (void)Idx;
            using E = std::remove_reference_t<decltype(elem)>;
            if constexpr (!detail::is_copy_tile_op_v<E>) {
                elem.init();
            }
        });
    }

    // -----------------------------------------------------------------
    // Phase 3: per-tile loop.
    // -----------------------------------------------------------------
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // 3a. Cumulative-wait shapes: emit growing cb_wait_front before consume.
        detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
            using E = std::remove_reference_t<decltype(elem)>;
            if constexpr (detail::is_copy_tile_op_v<E>) {
                constexpr bool is_first = detail::is_first_user_of_cb<Idx, TupleT>();
                if constexpr (is_first) {
                    detail::run_cumulative_wait(elem, i);
                }
            }
        });

        tile_regs_acquire();

        // 3b. Run each chain element in declared order.
        detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
            using E = std::remove_reference_t<decltype(elem)>;
            if constexpr (detail::is_copy_tile_op_v<E>) {
                constexpr bool is_first = detail::is_first_user_of_cb<Idx, TupleT>();
                constexpr bool is_last = detail::is_last_user_of_cb<Idx, TupleT>();
                // RunInit = true when not hoisting OR when chain is FPU-clashing
                // (which forces per-iter reinit even when the user opted in).
                constexpr bool run_init = !do_hoist;
                detail::run_copy_tile<E, run_init>(elem, i, is_first, is_last);
            } else {
                if constexpr (!do_hoist) {
                    elem.init();
                }
                elem.exec();
            }
        });

        tile_regs_commit();
        tile_regs_wait();

        // 3c. Pack the output slot. The chain's "result" lives in the highest
        // logical out — but for simple unary chains it's slot 0 (the CopyTile
        // target). We pack from slot 0 by convention; multi-output chains
        // (post-V1) need to extend this.
        cb_reserve_back(OutCB, 1);
        pack_tile(0, OutCB);
        cb_push_back(OutCB, 1);

        tile_regs_release();
    }

    // -----------------------------------------------------------------
    // Phase 4: upfront pops for upfront / cumulative policies.
    // -----------------------------------------------------------------
    detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
        using E = std::remove_reference_t<decltype(elem)>;
        if constexpr (detail::is_copy_tile_op_v<E>) {
            constexpr bool is_first = detail::is_first_user_of_cb<Idx, TupleT>();
            if constexpr (is_first) {
                detail::run_upfront_pop(elem, num_tiles);
            }
        }
    });

    // -----------------------------------------------------------------
    // Phase 5: reset CopyTile internal state.
    // -----------------------------------------------------------------
    detail::for_each_indexed(chain.elements, [&]<std::size_t Idx>(auto& elem) {
        (void)Idx;
        using E = std::remove_reference_t<decltype(elem)>;
        if constexpr (detail::is_copy_tile_op_v<E>) {
            detail::reset_runtime_state(elem);
        }
    });
}

}  // namespace compute_kernel_lib
