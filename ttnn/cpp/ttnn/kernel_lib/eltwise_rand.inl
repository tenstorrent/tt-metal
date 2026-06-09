// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_rand.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/rand.h"

namespace compute_kernel_lib {

/// RandTile chain element.
///
/// The caller must call `rand_tile_init(seed)` ONCE outside the chain (typically right
/// after `init_sfpu`); the element's `init()` is a no-op because the chain can't thread a
/// runtime seed through the static init hook (same pattern as `Dropout<Slot>`). The
/// per-instance `from` / `scale` (uniform [from, from+scale] range) is passed at construction.
///
/// @code
///   rand_tile_init(get_arg_val<uint32_t>(0));     // out-of-band, once
///   eltwise_chain(num_tiles,
///       RandTile<Dst::D0>{from, scale},
///       PackTile<cb_out, Dst::D0, OutStreaming, PackTileReconfig::None>{});
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
