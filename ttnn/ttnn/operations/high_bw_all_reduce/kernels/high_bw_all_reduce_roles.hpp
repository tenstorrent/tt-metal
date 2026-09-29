// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — per-chunk chain roles (single source for every kernel).
//
// The device sits at position `pos` of an ordered group path of `group_size` devices. Lane chunk
// c = k * W + r is block k of reducer r. Each chunk belongs to a slice; slice j's chain starts
// (head) at position j and ends (tail) at position j-1 (mod group_size). Partials always travel
// pos -> pos+1, finals pos -> pos-1.
//   * R1 / R2 line (num_slices = 1): head = position 0, tail = position G-1 for every chunk.
//   * R3 rotated-chain ring (num_slices = G): the head rotates with the slice, so every link —
//     including the closing edge — carries partials one way and finals the other, (G-1)/G of the
//     chunks per direction.
//
// Slice assignment (ring): reducer r's block stream is cut into n_seg = G / gcd(G, W) contiguous
// segments; segment i belongs to slice (r + i*W) mod G. Every slice gets the same share (W/gcd
// (reducer, segment) pairs), and inside a segment a reducer's stream is a plain line chain, so the
// per-reducer pipelines never couple around the ring (a head that rotated chunk by chunk would
// make every hop wait on the previous device's previous chunk — a latency cycle per chunk). At
// W = G (2x2 snake ring, W = 4) every reducer serves exactly one slice.

#pragma once

#include <cstdint>

template <uint32_t pos, uint32_t group_size, uint32_t num_slices>
struct ChainRoles {
    static_assert(num_slices == 1 || num_slices == group_size, "line (1 slice) or ring (G slices)");

    uint32_t num_reducers;
    uint32_t num_segments;

    explicit ChainRoles(uint32_t w) : num_reducers(w), num_segments(1) {
        if constexpr (num_slices > 1) {
            uint32_t a = group_size, b = w;
            while (b != 0) {
                const uint32_t t = a % b;
                a = b;
                b = t;
            }
            num_segments = group_size / a;
        }
    }

    // Blocks reducer r owns in a lane of num_chunks chunks.
    uint32_t blocks_of(uint32_t r, uint32_t num_chunks) const {
        return num_chunks > r ? (num_chunks - r + num_reducers - 1) / num_reducers : 0;
    }

    // Head position of block k (of nb) of reducer r.
    uint32_t head_of(uint32_t r, uint32_t k, uint32_t nb) const {
        if constexpr (num_slices == 1) {
            return 0;
        } else {
            const uint32_t seg = (k * num_segments) / nb;
            return (r + num_reducers * seg) % group_size;
        }
    }
    // head: no upstream partial (compute copies); final arrives from pos+1, is not forwarded.
    bool is_head(uint32_t r, uint32_t k, uint32_t nb) const { return head_of(r, k, nb) == pos; }
    // tail: the local sum is final; the reducer writes DRAM and hands the final to port_bwd.
    bool is_tail(uint32_t r, uint32_t k, uint32_t nb) const {
        return (head_of(r, k, nb) + group_size - 1) % group_size == pos;
    }
};
