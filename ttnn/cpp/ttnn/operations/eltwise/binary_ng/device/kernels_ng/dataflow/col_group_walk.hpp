// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>

// Walks a channel in column groups: `group` columns row by row, then the next group.
struct ColGroupWalk {
    uint32_t Ht, Wt, group, last_group;
    uint32_t g, width, th, k;

    ColGroupWalk(uint32_t start_in_channel, uint32_t Ht, uint32_t Wt, uint32_t group) :
        Ht(Ht), Wt(Wt), group(group), last_group((Wt - 1) / group) {
        const uint32_t group_tiles = Ht * group;
        g = std::min(start_in_channel / group_tiles, last_group);
        width = std::min(group, Wt - g * group);
        const uint32_t j = start_in_channel - g * group_tiles;
        th = j / width;
        k = j % width;
    }

    uint32_t tw() const { return g * group + k; }

    // Next tile; true when a new channel starts.
    bool next() {
        if (++k < width) {
            return false;
        }
        k = 0;
        if (++th < Ht) {
            return false;
        }
        th = 0;
        const bool next_channel = ++g > last_group;
        if (next_channel) {
            g = 0;
        }
        width = std::min(group, Wt - g * group);
        return next_channel;
    }
};
