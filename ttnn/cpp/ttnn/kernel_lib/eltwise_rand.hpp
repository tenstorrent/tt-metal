// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_rand.hpp
 * @brief Rand chain element — RandTile.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/rand.h"

namespace compute_kernel_lib {

/// RandTile chain element.
///
/// Runtime seed: the caller must invoke `rand_tile_init(seed)` ONCE outside
/// the chain (typically right after `init_sfpu` / `unary_op_init_common`)
/// with the runtime-supplied seed. The chain's element `init()` is a no-op
/// — same pattern as `Dropout<Slot>` (eltwise_scalar.hpp).
///
/// Per-instance runtime payload covers `from` / `scale` (the uniform
/// [from, from + scale] range), passed at construction.
///
/// Usage:
/// @code
///   uint32_t seed = get_arg_val<uint32_t>(0);
///   rand_tile_init(seed);                         // out-of-band, once
///   eltwise_chain(num_tiles,
///       RandTile<Dst::D0>{from, scale},           // per-tile rand
///       PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
/// @endcode
template <Dst DstSlot>
struct RandTile : RandTileTag, UnaryOp<RandTile<DstSlot>, DstSlot> {
    /// Runtime payload — `from` and `scale` define the uniform [from, from+scale] range.
    uint32_t from_;
    uint32_t scale_;

    constexpr RandTile(uint32_t f, uint32_t s) noexcept : from_(f), scale_(s) {}
    constexpr RandTile() noexcept : from_(0), scale_(0) {}

    /// No-op: `rand_tile_init(seed)` MUST be called by the caller with the
    /// runtime seed before the first `eltwise_chain(...)` invocation that
    /// contains a RandTile element. The chain has no way to thread a runtime
    /// seed through the static `init()` hook.
    static ALWI void init() {}
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        rand_tile(to_u32(DstSlot) + slot_offset, from_, scale_);
    }
};

}  // namespace compute_kernel_lib
