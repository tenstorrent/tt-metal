// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_fill.hpp
 * @brief Fill chain elements — FillScalar, FillInt, FillBitcast.
 *
 * These elements write a constant into a DEST slot. They derive `FillTileTag` (rooted in
 * `DestOnlyTag`) so trait sweeps that look at CB consumers / CB producers correctly skip them.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/fill.h"

namespace compute_kernel_lib {

template <Dst DstSlot>
struct FillScalar : FillTileTag, UnaryOp<FillScalar<DstSlot>, DstSlot> {
    float value;
    constexpr explicit FillScalar(float v) noexcept : value(v) {}
    constexpr FillScalar() noexcept : value(0.0f) {}

    static ALWI void init() { fill_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { fill_tile(to_u32(DstSlot), value); }
};

template <DataFormat DF, Dst DstSlot>
struct FillInt : FillTileTag, UnaryOp<FillInt<DF, DstSlot>, DstSlot> {
    uint32_t value;
    constexpr explicit FillInt(uint32_t v) noexcept : value(v) {}
    constexpr FillInt() noexcept : value(0) {}

    // fill_tile_int shares the same init as fill_tile (no separate `_int_init`).
    static ALWI void init() { fill_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { fill_tile_int<DF>(to_u32(DstSlot), value); }
};

template <Dst DstSlot>
struct FillBitcast : FillTileTag, UnaryOp<FillBitcast<DstSlot>, DstSlot> {
    uint32_t bits;
    constexpr explicit FillBitcast(uint32_t b) noexcept : bits(b) {}
    constexpr FillBitcast() noexcept : bits(0) {}

    static ALWI void init() { fill_tile_init(); }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const { fill_tile_bitcast(to_u32(DstSlot), bits); }
};

}  // namespace compute_kernel_lib
