// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of rand.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/rand.h"

namespace compute_kernel_lib {

/// 32-bit finalizer shared by the host key derivation and the in-kernel salt/epoch folding.
ALWI uint32_t rand_mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

/// RandTile chain element.
///
/// `rand_tile_init(seed, stream_id)` is rand's per-op SFPU init (it seeds and domain-separates the PRNG) — the rand
/// analogue of every other op's `*_tile_init`, except it takes the runtime seed. So
/// `init()` is a normal instance method (non-static, like `exec`) that reads the
/// element's `seed_` and `stream_id_`; the chain dispatches init on the instance, emitting it once at
/// boot-hoist — the once-per-kernel seeding the kernel used to do out-of-band. The
/// per-instance `from` / `scale` (uniform [from, from+scale] range), `seed`, and `stream_id` are all
/// passed at construction (same pattern as `Dropout<Slot>`). A nonzero `salt_key` folds a per-tile
/// value into the SFPU lane salt so equal PRNG states at different tiles yield different outputs.
///
/// @code
///   eltwise_chain(IterationShape::tiles(num_tiles),
///       RandTile<Dst::D0>{from, scale, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1)},
///       PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
/// @endcode
template <Dst DstSlot>
struct RandTile : RandTileTag, UnaryOp<RandTile<DstSlot>, DstSlot> {
    uint32_t from_;
    uint32_t scale_;
    uint32_t seed_;
    uint32_t stream_id_;
    uint32_t salt_key_;

    constexpr RandTile(uint32_t f, uint32_t s, uint32_t seed, uint32_t stream_id = 0, uint32_t salt_key = 0) noexcept :
        from_(f), scale_(s), seed_(seed), stream_id_(stream_id), salt_key_(salt_key) {}
    constexpr RandTile() noexcept : from_(0), scale_(0), seed_(0), stream_id_(0), salt_key_(0) {}

    ALWI void init() const { rand_tile_init(seed_, stream_id_); }
    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
        const uint32_t salt = salt_key_ != 0 ? rand_mix32(salt_key_ + i * 0x9E3779B9u) : 0u;
        rand_tile(to_u32(DstSlot) + slot_offset, from_, scale_, salt);
    }
};

/// ThreefryTile chain element: Threefry-2x32 keyed by `(key0, key1)`, counter = global tile index.
/// Output depends only on key and position, so it is identical across core grids and work splits.
template <Dst DstSlot>
struct ThreefryTile : ThreefryTileTag, UnaryOp<ThreefryTile<DstSlot>, DstSlot> {
    static constexpr int rounds = 20;
    static constexpr uint32_t counters_per_tile = 16;

    uint32_t from_;
    uint32_t scale_;
    uint32_t key0_;
    uint32_t key1_;
    uint32_t start_id_;

    constexpr ThreefryTile(uint32_t f, uint32_t s, uint32_t key0, uint32_t key1, uint32_t start_id = 0) noexcept :
        from_(f), scale_(s), key0_(key0), key1_(key1), start_id_(start_id) {}
    constexpr ThreefryTile() noexcept : from_(0), scale_(0), key0_(0), key1_(0), start_id_(0) {}

    ALWI void init() const { threefry_tile_init(); }
    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
        threefry_tile<rounds>(
            to_u32(DstSlot) + slot_offset, from_, scale_, key0_, key1_, (start_id_ + i) * counters_per_tile);
    }
};

}  // namespace compute_kernel_lib
