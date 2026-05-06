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

template <Dst DstSlot>
struct RandTile : RandTileTag, UnaryOp<RandTile<DstSlot>, DstSlot> {
    /// Runtime payload — `from` and `scale` define the uniform [from, from+scale] range.
    uint32_t from_;
    uint32_t scale_;
    /// Optional seed (passed to `rand_tile_init`). Default 0.
    uint32_t seed_;

    constexpr RandTile(uint32_t f, uint32_t s, uint32_t seed = 0) noexcept : from_(f), scale_(s), seed_(seed) {}
    constexpr RandTile() noexcept : from_(0), scale_(0), seed_(0) {}

    ALWI void init() const { rand_tile_init(seed_); }  // seed is runtime-bound
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { rand_tile(to_u32(DstSlot), from_, scale_); }
};

}  // namespace compute_kernel_lib
