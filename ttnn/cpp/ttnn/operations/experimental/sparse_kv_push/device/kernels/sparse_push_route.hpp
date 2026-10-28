// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Route choice shared by the push reader and writer, which must agree: the writer's done increments must follow the
// reader's data. The mesh is a 2D torus, so every row and column is a ring. On a ring of even size n, a chip n / 2 hops
// away is equally far both ways (a tie), and the fabric routing tables always break ties the same way (north, east),
// overloading those links while south and west idle. route() sends half of the tie destinations the other way round.

namespace sparse_push {

constexpr uint32_t kEast = 0, kWest = 1, kNorth = 2, kSouth = 3;
constexpr uint8_t kActionEast = 1, kActionWest = 2;  // X-map action bytes in a fabric packet header

// N <-> S; any other direction unchanged.
constexpr uint32_t flip_north_south(uint32_t dir) { return dir == kNorth ? kSouth : (dir == kSouth ? kNorth : dir); }

struct Route {
    uint32_t dir;    // outgoing connection direction
    bool flip_turn;  // flip the E/W action byte at the turn column (see flip_turn())
};

// Route from chip src_chip to chip dst_chip (row-major chip indices), given the routing table's first-hop direction.
template <uint32_t rows, uint32_t cols>
inline Route route(uint32_t src_chip, uint32_t dst_chip, uint32_t next_hop_dir) {
    const uint32_t src_row = src_chip / cols, src_col = src_chip % cols;
    const uint32_t dst_row = dst_chip / cols, dst_col = dst_chip % cols;
    // Hops towards higher row / column index, around each ring.
    const uint32_t dy = (dst_row + rows - src_row) % rows, dx = (dst_col + cols - src_col) % cols;
    const bool row_tie = rows % 2 == 0 && dy == rows / 2;
    const bool col_tie = cols % 2 == 0 && dx == cols / 2;
    // Routes go Y then X, so a row tie decides the first hop (N or S). Flip the tie destinations whose column parity
    // differs from ours: half of the tie row.
    if (row_tie && ((src_col + dst_col) & 1u)) {
        next_hop_dir = flip_north_south(next_hop_dir);
    }
    // A column tie decides the E/W turn into the destination row, which the header's X map encodes: flip the tie
    // destinations whose row parity differs from ours. Same-row ties (dy == 0) never meet this rule
    // (src_row + dst_row is even), so they keep the table's east.
    if (col_tie && ((src_row + dst_row) & 1u)) {
        return {next_hop_dir, true};
    }
    return {next_hop_dir, false};
}

// Applies Route::flip_turn to a header already filled by fabric_set_unicast_route. The header carries the
// destination's action maps, Y at route_buffer[0, rows) and X at route_buffer[rows, rows + cols), one byte per row or
// column saying which way to go from there. Packets turn at the source column, the only column at the exact tie
// distance (every other column has one shortest way), so flipping that one byte reroutes the whole packet.
template <uint32_t rows, uint32_t cols, typename Header>
inline void flip_turn(volatile Header* hdr, uint32_t src_chip) {
    volatile uint8_t& action = hdr->route_buffer[rows + src_chip % cols];
    action = action == kActionEast ? kActionWest : (action == kActionWest ? kActionEast : action);
}

}  // namespace sparse_push
