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
/// `Seed` is a compile-time NTTP. Per-instance runtime payload covers `from` / `scale`
/// (the uniform [from, from+scale] range). Overrides `exec(uint32_t)` directly.
template <Dst DstSlot, uint32_t Seed>
struct RandTile : RandTileTag, UnaryOp<RandTile<DstSlot, Seed>, DstSlot> {
    /// Runtime payload — `from` and `scale` define the uniform [from, from+scale] range.
    uint32_t from_;
    uint32_t scale_;

    constexpr RandTile(uint32_t f, uint32_t s) noexcept : from_(f), scale_(s) {}
    constexpr RandTile() noexcept : from_(0), scale_(0) {}

    static ALWI void init() { rand_tile_init(Seed); }  // seed is compile-time NTTP
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        rand_tile(to_u32(DstSlot) + slot_offset, from_, scale_);
    }
};

}  // namespace compute_kernel_lib
