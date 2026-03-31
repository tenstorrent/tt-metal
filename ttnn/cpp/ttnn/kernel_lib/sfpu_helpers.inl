// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file sfpu_helpers.inl
 * @brief Implementation of SFPU pipeline and convenience functions
 *
 * This file contains the implementation details for sfpu_pipeline(), sfpu_op(),
 * and the named convenience aliases. It should only be included by sfpu_helpers.hpp.
 */

#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/xielu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

constexpr bool sfpu_reconfig_input(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::INPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

constexpr bool sfpu_reconfig_output(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::OUTPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

/**
 * @brief Loader functor used by sfpu_pipeline to handle Load ops
 *
 * Tracks the last CB used for _with_dt reconfiguration. On first call,
 * initializes with copy_tile_to_dst_init_short. On subsequent calls with
 * a different CB, uses copy_tile_to_dst_init_short_with_dt.
 *
 * For WaitAndPopPerTile: deduplicates wait/pop per unique CB. When two Loads
 * reference the same CB (e.g., Load<cb, D0>, Load<cb, D1>), waits once before
 * the first copy and defers pop until all copies from that CB are complete.
 */
template <SfpuInputPolicy input_policy>
struct TileLoader {
    uint32_t last_cb;
    bool initialized;
    uint32_t tile_idx;  // 0 for streaming (WaitAndPopPerTile), i for indexed

    // CB dedup tracking for wait/pop (max 8 unique CBs)
    static constexpr uint32_t MAX_CBS = 8;
    uint32_t waited_cbs[MAX_CBS];
    uint32_t num_waited;

    template <typename LoadOp>
    ALWI void operator()(const LoadOp& load) {
        constexpr uint32_t cb = LoadOp::cb;
        constexpr uint32_t dst = LoadOp::dst_idx;

        // _with_dt tracking: reconfigure unpacker when CB changes
        if (!initialized) {
            copy_tile_to_dst_init_short(cb);
            last_cb = cb;
            initialized = true;
        } else if (cb != last_cb) {
            copy_tile_to_dst_init_short_with_dt(last_cb, cb);
            last_cb = cb;
        }

        // CB synchronization: wait once per unique CB
        if constexpr (input_policy == SfpuInputPolicy::WaitAndPopPerTile) {
            bool already_waited = false;
            for (uint32_t i = 0; i < num_waited; ++i) {
                if (waited_cbs[i] == cb) { already_waited = true; break; }
            }
            if (!already_waited) {
                cb_wait_front(cb, 1);
                waited_cbs[num_waited++] = cb;
            }
        }

        // Copy tile from CB to DEST slot
        // For streaming: tile_idx is always 0 (tile stays at CB front until all copies done)
        // For upfront/no-wait: tile_idx increments through the CB
        copy_tile(cb, tile_idx, dst);
    }

    // Pop all waited CBs (called after all Loads are complete)
    ALWI void pop_all() {
        if constexpr (input_policy == SfpuInputPolicy::WaitAndPopPerTile) {
            for (uint32_t i = 0; i < num_waited; ++i) {
                cb_pop_front(waited_cbs[i], 1);
            }
        }
    }
};

/**
 * @brief Functor that reconfigures unpacker format to the first Load's CB
 *
 * Only reconfigures once — on the first Load's CB. This is a safety net for
 * when sfpu_pipeline is called after a different operation (reduce, binary FPU)
 * that may have left the unpacker in an unknown state. Within the tile loop,
 * TileLoader's copy_tile_to_dst_init_short handles the full unpacker init,
 * and _with_dt handles subsequent CB switches conditionally.
 */
struct InputReconfigFunctor {
    bool done;
    template <typename LoadOp>
    ALWI void operator()(const LoadOp&) {
        if (!done) {
            reconfig_data_format_srca(LoadOp::cb);
            done = true;
        }
    }
};

/** @brief Functor that waits for num_tiles on each Load's CB */
struct UpfrontWaitFunctor {
    uint32_t num_tiles;
    template <typename LoadOp>
    ALWI void operator()(const LoadOp&) {
        cb_wait_front(LoadOp::cb, num_tiles);
    }
};

}  // namespace detail

// =============================================================================
// Pipeline Implementation
// =============================================================================

template <
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

    // Runtime DEST capacity validation
    ASSERT(static_cast<uint32_t>(pack_slot) < DEST_AUTO_LIMIT);

    // Data format reconfiguration (once before the tile loop)
    // Input: reconfig unpacker to first Load's CB format. This is a safety net
    // for when the pipeline follows a different operation type. Within the tile
    // loop, TileLoader handles CB switches via copy_tile_to_dst_init_short/_with_dt.
    if constexpr (detail::sfpu_reconfig_input(reconfig)) {
        detail::InputReconfigFunctor reconfig_fn{false};
        chain.for_each_load(reconfig_fn);
    }
    // Output: reconfig packer to output CB format.
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Upfront waits for non-streaming input policies
    if constexpr (input_policy == SfpuInputPolicy::WaitUpfrontNoPop) {
        // Wait for num_tiles on each Load's CB. If multiple Loads reference
        // the same CB, the redundant cb_wait_front calls are harmless (idempotent).
        detail::UpfrontWaitFunctor wait_fn{num_tiles};
        chain.for_each_load(wait_fn);
    }

    // Bulk output: reserve all tiles upfront
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    // Per-tile streaming loop
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        // --- Load phase ---
        // Tile index: 0 for streaming (WaitAndPopPerTile pops after each),
        // i for indexed access (WaitUpfrontNoPop/NoWaitNoPop)
        constexpr bool streaming = (input_policy == SfpuInputPolicy::WaitAndPopPerTile);
        const uint32_t tile_idx = streaming ? 0 : i;

        detail::TileLoader<input_policy> loader{0, false, tile_idx, {}, 0};
        chain.for_each_load(loader);
        loader.pop_all();

        // --- Compute phase ---
        chain.apply();

        // --- Pack phase ---
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

    // Bulk output: push all tiles at end
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_push_back(ocb, num_tiles);
    }
}

// =============================================================================
// Convenience: Single Unary Op Implementation
// =============================================================================

template <
    uint32_t ICB,
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op) {
    auto chain = sfpu_chain(Load<ICB, Dst::D0>{}, op);
    sfpu_pipeline<input_policy, output_policy, reconfig>(chain, ocb, num_tiles);
}

// =============================================================================
// Named Convenience Aliases Implementation
// =============================================================================

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Exp<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Log<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Log1p<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Sqrt<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Rsqrt<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Recip<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Abs<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Neg<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Sigmoid<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Tanh<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Gelu<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Silu<>{});
}

template <uint32_t ICB, SfpuInputPolicy input_policy, SfpuOutputPolicy output_policy, SfpuDataFormatReconfig reconfig>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, input_policy, output_policy, reconfig>(ocb, num_tiles, Relu<>{});
}

}  // namespace compute_kernel_lib
