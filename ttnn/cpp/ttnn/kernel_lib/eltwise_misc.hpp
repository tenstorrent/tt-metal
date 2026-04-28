// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rand.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_misc.hpp
 * @brief Tier 2 misc: Identity, Typecast, FillTile, FillTileBitcast, RandTile.
 *
 * Note: `FillScalar` and `FillConst` already live in `eltwise_chain.hpp` as
 * core elements; they are the chain-element form of `fill_tile`. The structs
 * below mirror them for API completeness when callers prefer the float-typed
 * field name (`fill_val`) or do not need them in a chain.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Dst Slot = Dst::D0>
struct Identity : UnaryOp<Identity<Slot>, Slot> {
    ALWI void init() const { identity_tile_init(); }
    ALWI void call(uint32_t d) const { identity_tile(d); }
};

template <uint32_t InDtype, uint32_t OutDtype, Dst Slot = Dst::D0>
struct Typecast : UnaryOp<Typecast<InDtype, OutDtype, Slot>, Slot> {
    ALWI void init() const { typecast_tile_init<InDtype, OutDtype>(); }
    ALWI void call(uint32_t d) const { typecast_tile<InDtype, OutDtype>(d); }
};

/// Float-typed fill (alternative ergonomics to `FillScalar`).
template <Dst Slot = Dst::D0>
struct FillTile : UnaryOp<FillTile<Slot>, Slot> {
    float fill_val;
    ALWI void init() const { fill_tile_init(); }
    ALWI void call(uint32_t d) const { fill_tile(d, fill_val); }
};

/// Bitcast fill — runtime uint32_t bit pattern interpreted as the target
/// element type (float / int).
template <Dst Slot = Dst::D0>
struct FillTileBitcast : UnaryOp<FillTileBitcast<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { fill_tile_init(); }
    ALWI void call(uint32_t d) const { fill_tile_bitcast(d, param0); }
};

/// Per-tile random fill (uniform in `[from, from + scale)` after bit-pattern
/// decode). Init is owned by the caller (rand seeding lives outside the chain).
template <Dst Slot = Dst::D0>
struct RandTile : UnaryOp<RandTile<Slot>, Slot> {
    uint32_t from;
    uint32_t scale;
    ALWI void init() const { /* no per-tile init */ }
    ALWI void call(uint32_t d) const { rand_tile(d, from, scale); }
};

}  // namespace compute_kernel_lib::eltwise
