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
    bool rotate_waves;
};

inline constexpr uint32_t wave_column_shift(uint32_t wave, uint32_t wave_count, uint32_t cols) {
    const uint32_t stride = cols / (wave_count + 1) > 0 ? cols / (wave_count + 1) : 1;
    return ((wave + wave / 2) * stride) % cols;
}

// Count a lane's work without expanding its band list. Each wave assigns a
// strided subset of units and contributes either one or two physical shards.
inline constexpr uint32_t band_count(const Geometry& g, uint32_t lane) {
    const uint32_t wave_count = g.ring_size / 2 + 1;
    const uint32_t lane_count = g.num_blocks * g.cols;
    uint32_t count = 0;
    for (uint32_t wave = 0; wave < wave_count; ++wave) {
        const bool paired = wave > 0 && 2 * wave < g.ring_size;
        const uint32_t shift = g.rotate_waves && paired ? wave_column_shift(wave, wave_count, g.cols) : 0;
        const uint32_t first = lane % g.num_blocks + ((lane / g.num_blocks + g.cols - shift) % g.cols) * g.num_blocks;
        if (first < g.units_per_shard) {
            count += (1 + (g.units_per_shard - 1 - first) / lane_count) * (paired ? 2 : 1);
        }
    }
    return count;
}

class Iterator {
public:
    constexpr Iterator(Geometry geometry, uint32_t lane) : geometry_(geometry), lane_(lane), unit_(lane) {}

    // Returns false at exhaustion, including empty lanes. get_shard indexes the
    // ring-arrival order after mapping transport ranks to tensor ranks.
    template <typename GetShard>
    constexpr bool next(GetShard get_shard, uint32_t& physical_start) {
        const uint32_t wave_count = geometry_.ring_size / 2 + 1;
        while (wave_ < wave_count) {
            const uint32_t first_shard = wave_ == 0 ? 0 : 2 * wave_ - 1;
            const uint32_t wave_size = wave_ == 0 || first_shard + 1 == geometry_.ring_size ? 1 : 2;
            if (unit_ < geometry_.units_per_shard) {
                physical_start =
                    get_shard(first_shard + member_) * geometry_.tiles_per_shard + unit_ * geometry_.tiles_per_unit;
                if (++member_ == wave_size) {
                    member_ = 0;
                    unit_ += geometry_.num_blocks * geometry_.cols;
                }
                return true;
            }
            ++wave_;
            const bool paired = wave_ > 0 && 2 * wave_ < geometry_.ring_size;
            const uint32_t shift =
                geometry_.rotate_waves && paired ? wave_column_shift(wave_, wave_count, geometry_.cols) : 0;
            const uint32_t column = lane_ / geometry_.num_blocks;
            unit_ = lane_ % geometry_.num_blocks +
                    ((column + geometry_.cols - shift) % geometry_.cols) * geometry_.num_blocks;
        }
        return false;
    }

private:
    Geometry geometry_;
    uint32_t lane_;
    uint32_t unit_;
    uint32_t wave_ = 0;
    uint32_t member_ = 0;
};

template <uint32_t RingSize, uint32_t KTiles, uint32_t UnitTiles, uint32_t Blocks, uint32_t Cols, bool Rotate>
constexpr Iterator for_lane(uint32_t lane) {
    constexpr uint32_t shard_tiles = KTiles / RingSize;
    return Iterator(
        {RingSize, (shard_tiles + UnitTiles - 1) / UnitTiles, shard_tiles, UnitTiles, Blocks, Cols, Rotate}, lane);
}

}  // namespace indexer_ring_schedule
