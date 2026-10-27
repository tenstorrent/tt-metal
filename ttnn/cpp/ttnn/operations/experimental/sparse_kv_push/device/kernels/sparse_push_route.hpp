// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the push reader and writer, which must agree. The routing tables break every ring-distance tie the same
// way (N, E), overloading those links, so half of the tie destinations are sent the other way round.

#pragma once

#include <cstdint>

namespace sparse_push {

constexpr uint32_t kEast = 0, kWest = 1, kNorth = 2, kSouth = 3;
constexpr uint8_t kActEast = 1, kActWest = 2;

struct Route {
    uint32_t dir;    // outgoing connection direction
    bool flip_turn;  // flip the E/W action byte at the turn column
};

template <uint32_t rows, uint32_t cols>
inline Route route(uint32_t src, uint32_t dst, uint32_t table_dir) {
    const uint32_t r = src / cols, c = src % cols, rd = dst / cols, cd = dst % cols;
    const uint32_t dy = (rd + rows - r) % rows, dx = (cd + cols - c) % cols;
    const bool row_tie = rows % 2 == 0 && dy == rows / 2;
    const bool col_tie = cols % 2 == 0 && dx == cols / 2;
    if (row_tie && ((c + cd) & 1u)) {
        table_dir = table_dir == kNorth ? kSouth : (table_dir == kSouth ? kNorth : table_dir);
    }
    if (col_tie && ((r + rd) & 1u)) {
        if (dy == 0) {
            return {table_dir == kEast ? kWest : (table_dir == kWest ? kEast : table_dir), false};
        }
        // Routes go Y then X, so off the source row the E/W choice is the turn at the source column.
        return {table_dir, true};
    }
    return {table_dir, false};
}

// Applies Route::flip_turn to a header already filled by fabric_set_unicast_route.
template <uint32_t rows, uint32_t cols, typename Header>
inline void flip_turn(volatile Header* hdr, uint32_t src) {
    volatile uint8_t& act = hdr->route_buffer[rows + src % cols];
    act = act == kActEast ? kActWest : (act == kActWest ? kActEast : act);
}

}  // namespace sparse_push
