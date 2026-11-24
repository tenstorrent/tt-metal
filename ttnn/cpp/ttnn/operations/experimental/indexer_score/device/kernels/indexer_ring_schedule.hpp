// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace indexer_ring_schedule {

// Pure host/device iterator over one lane's shard-major K units. The shard order is
// shared by every core; only lane = block + column * num_blocks is per-core.
// Geometry is compile-time constant in kernels, without specializing on core/rank.
struct Geometry {
    uint32_t ring_size;
    uint32_t units_per_shard;
    uint32_t tiles_per_shard;
    uint32_t tiles_per_unit;
    uint32_t num_blocks;
    uint32_t cols;
};

// Global round-robin over arrival order: every (wave, unit, shard-of-wave) item gets one running index and
// lane l takes the items whose index is l mod lane_count. Each wave starts where the previous one stopped, so
// the lanes share the remainders of all waves and the busiest lane carries ceil(items / lanes) units. Restarting
// at lane 0 every wave instead piled each wave's remainder onto the same lanes. Which core computes a unit
// moves; how it is computed does not, so the output is unchanged.
constexpr uint32_t wave_items(const Geometry& g, uint32_t wave) {
    const uint32_t first_shard = wave == 0 ? 0 : 2 * wave - 1;
    const uint32_t wave_size = wave == 0 || first_shard + 1 == g.ring_size ? 1 : 2;
    return g.units_per_shard * wave_size;
}

constexpr uint32_t band_count(const Geometry& g, uint32_t lane) {
    const uint32_t wave_count = g.ring_size / 2 + 1;
    const uint32_t lane_count = g.num_blocks * g.cols;
    uint32_t count = 0, base = 0;
    for (uint32_t wave = 0; wave < wave_count; ++wave) {
        const uint32_t n = wave_items(g, wave);
        const uint32_t start = (lane + lane_count - base % lane_count) % lane_count;
        if (start < n) {
            count += 1 + (n - 1 - start) / lane_count;
        }
        base += n;
    }
    return count;
}

class Iterator {
public:
    constexpr Iterator(Geometry geometry, uint32_t lane) : geometry_(geometry), lane_(lane), idx_(lane) {}

    // Returns false at exhaustion, including empty lanes. get_shard indexes the
    // ring-arrival order after mapping transport ranks to tensor ranks.
    template <typename GetShard>
    constexpr bool next(GetShard get_shard, uint32_t& physical_start) {
        const uint32_t wave_count = geometry_.ring_size / 2 + 1;
        const uint32_t lane_count = geometry_.num_blocks * geometry_.cols;
        while (wave_ < wave_count) {
            const uint32_t first_shard = wave_ == 0 ? 0 : 2 * wave_ - 1;
            const uint32_t wave_size = wave_ == 0 || first_shard + 1 == geometry_.ring_size ? 1 : 2;
            const uint32_t n = geometry_.units_per_shard * wave_size;
            if (idx_ < n) {
                const uint32_t unit = idx_ / wave_size;
                const uint32_t member = idx_ - unit * wave_size;
                physical_start =
                    get_shard(first_shard + member) * geometry_.tiles_per_shard + unit * geometry_.tiles_per_unit;
                idx_ += lane_count;
                return true;
            }
            base_ += n;
            ++wave_;
            idx_ = (lane_ + lane_count - base_ % lane_count) % lane_count;
        }
        return false;
    }

private:
    Geometry geometry_;
    uint32_t lane_;
    uint32_t idx_;
    uint32_t base_ = 0;
    uint32_t wave_ = 0;
};

template <uint32_t RingSize, uint32_t KTiles, uint32_t UnitTiles, uint32_t Blocks, uint32_t Cols>
constexpr Iterator for_lane(uint32_t lane) {
    constexpr uint32_t shard_tiles = KTiles / RingSize;
    return Iterator({RingSize, (shard_tiles + UnitTiles - 1) / UnitTiles, shard_tiles, UnitTiles, Blocks, Cols}, lane);
}

}  // namespace indexer_ring_schedule
