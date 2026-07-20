// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_rand.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/rand.h"

namespace compute_kernel_lib {

/// RandTile chain element.
///
/// `rand_tile_init(seed)` is rand's per-op SFPU init (it seeds the PRNG) — the rand
/// analogue of every other op's `*_tile_init`, except it takes the runtime seed. So
/// `init()` is a normal instance method (non-static, like `exec`) that reads the
/// element's `seed_`; the chain dispatches init on the instance, emitting it once at
/// boot-hoist — the once-per-kernel seeding the kernel used to do out-of-band. The
/// per-instance `from` / `scale` (uniform [from, from+scale] range) and `seed` are all
/// passed at construction (same pattern as `Dropout<Slot>`).
///
/// @code
///   eltwise_chain(EltwiseShape::tiles(num_tiles),
///       RandTile<Dst::D0>{from, scale, get_arg_val<uint32_t>(0)},
///       PackTile<cb_out, output(OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
/// @endcode
template <Dst DstSlot>
struct RandTile : RandTileTag, UnaryOp<RandTile<DstSlot>, DstSlot> {
    /// Runtime payload — `from`/`scale` define the uniform [from, from+scale] range;
    /// `seed` seeds the PRNG once at init.
    uint32_t from_;
    uint32_t scale_;
    uint32_t seed_;

    constexpr RandTile(uint32_t f, uint32_t s, uint32_t seed) noexcept : from_(f), scale_(s), seed_(seed) {}
    constexpr RandTile() noexcept : from_(0), scale_(0), seed_(0) {}

    ALWI void init() const { rand_tile_init(seed_); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        rand_tile(to_u32(DstSlot) + slot_offset, from_, scale_);
    }
};

}  // namespace compute_kernel_lib
