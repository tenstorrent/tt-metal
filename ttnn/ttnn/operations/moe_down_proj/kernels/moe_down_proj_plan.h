// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj plan: the counts-derived row split, identical on all three kernels of every core (plan_wave).
// Pure integer code. Tile rows are expert-sorted regions of ceil(counts[e] / 32) tiles; the active rows
// [0, active_row_tiles) are cut into num_waves waves, and each wave into num_m_groups contiguous per-group ranges
// of core_row_tiles rows (the last ones shorter or empty). Every kernel walks the same channel-round order:
// per wave, per K round, the x round, then one w chunk per in-wave expert in expert order.

#pragma once

#include <cstdint>

constexpr uint32_t MOE_NUM_EXPERTS = 8;

struct MoePlan {
    // Region e = tile rows [region_start[e], region_start[e + 1]), clamped to active_row_tiles.
    uint32_t region_start[MOE_NUM_EXPERTS + 1];
    uint32_t active_row_tiles;
    uint32_t wave_row_tiles;
    uint32_t core_row_tiles;
    uint32_t num_m_groups;

    MoePlan(const uint32_t* counts, uint32_t tensor_row_tiles, uint32_t num_waves, uint32_t num_m_groups_) {
        num_m_groups = num_m_groups_;
        uint32_t total = 0;
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            region_start[e] = total;
            total += (counts[e] + 31) / 32;
        }
        region_start[MOE_NUM_EXPERTS] = total;
        // Malformed counts must never write past y.
        active_row_tiles = total < tensor_row_tiles ? total : tensor_row_tiles;
        for (uint32_t e = 0; e <= MOE_NUM_EXPERTS; ++e) {
            if (region_start[e] > active_row_tiles) {
                region_start[e] = active_row_tiles;
            }
        }
        wave_row_tiles = (active_row_tiles + num_waves - 1) / num_waves;
        core_row_tiles = (wave_row_tiles + num_m_groups - 1) / num_m_groups;
    }

    uint32_t wave_start(uint32_t wave) const {
        const uint32_t start = wave * wave_row_tiles;
        return start < active_row_tiles ? start : active_row_tiles;
    }
    uint32_t wave_end(uint32_t wave) const { return wave_start(wave + 1); }
    bool wave_empty(uint32_t wave) const { return wave_start(wave) == wave_end(wave); }

    // First tile row of group `group` in `wave`, and its row count (0 for an empty group).
    uint32_t group_first(uint32_t wave, uint32_t group) const {
        const uint32_t first = wave_start(wave) + group * core_row_tiles;
        const uint32_t end = wave_end(wave);
        return first < end ? first : end;
    }
    uint32_t group_rows(uint32_t wave, uint32_t group) const {
        const uint32_t first = group_first(wave, group);
        const uint32_t last = first + core_row_tiles;
        const uint32_t end = wave_end(wave);
        return (last < end ? last : end) - first;
    }

    // Rows of expert e inside [lo, hi): returns the count, first row in *first.
    uint32_t expert_rows_in(uint32_t e, uint32_t lo, uint32_t hi, uint32_t* first) const {
        const uint32_t a = region_start[e] > lo ? region_start[e] : lo;
        const uint32_t b = region_start[e + 1] < hi ? region_start[e + 1] : hi;
        *first = a;
        return a < b ? b - a : 0;
    }

    // Experts whose rows the wave touches; the w stream delivers exactly these, in expert order.
    bool expert_in_wave(uint32_t wave, uint32_t e) const {
        uint32_t first;
        return expert_rows_in(e, wave_start(wave), wave_end(wave), &first) != 0;
    }
};
