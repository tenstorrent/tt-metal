// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Length-balanced ("lean") decode schedule.
//
// The work of one decode call is a list of units (batch b, kv head h, k chunk j), ordered by segment
// s = b * NKV + h and then by j. Segment s has n_b = ceil((cur_pos_b + 1) / chunk) units, so the list
// follows the per-user positions that live on device. Every core derives the same list from cur_pos
// and takes an equal contiguous slice [start(c), start(c + 1)) of it. A segment cut by slice
// boundaries is computed in pieces: the core holding the segment's first unit is its root, every
// later core sends its partial (l, m, o) to that root, which combines them and writes the output rows.
namespace lean {

constexpr uint32_t SKIP_USER = 0xFFFFFFFFu;

inline uint32_t chunks_for_pos(uint32_t cur_pos, uint32_t chunk_size) {
    return cur_pos == SKIP_USER ? 0 : (cur_pos + chunk_size) / chunk_size;
}

struct Schedule {
    uint32_t U;  // total units
    uint32_t N;  // participating cores

    uint32_t start(uint32_t c) const { return c >= N ? U : (c * U) / N; }
    bool nonempty(uint32_t c) const { return start(c) < start(c + 1); }
    // Core whose slice contains unit u (u < U).
    uint32_t core_of(uint32_t u) const {
        uint32_t c = (u * N) / U;
        if (c >= N) {
            c = N - 1;
        }
        while (c + 1 < N && start(c + 1) <= u) {
            ++c;
        }
        while (c > 0 && start(c) > u) {
            --c;
        }
        return c;
    }
    // Number of non-empty cores strictly between lo and hi.
    uint32_t nonempty_between(uint32_t lo, uint32_t hi) const {
        uint32_t n = 0;
        for (uint32_t c = lo + 1; c < hi; ++c) {
            n += nonempty(c) ? 1 : 0;
        }
        return n;
    }
};

struct Piece {
    uint32_t b;
    uint32_t h;
    uint32_t j0;    // first chunk of this piece
    uint32_t j1;    // one past the last chunk
    uint32_t n;     // chunks in the whole segment
    bool is_root;   // holds chunk 0 but not the whole segment: combines children, writes output
    bool is_child;  // does not hold chunk 0: sends its partial to the root
    uint32_t root_core;
    uint32_t child_slot;
    uint32_t num_children;
};

template <uint32_t B, uint32_t NKV, typename GetN>
Schedule make_schedule(uint32_t num_cores, GetN get_n) {
    uint32_t U = 0;
    for (uint32_t b = 0; b < B; ++b) {
        U += get_n(b) * NKV;
    }
    return Schedule{U, num_cores};
}

// Walks this core's pieces in list order. Kernels keep their hot loop in kernel_main around next():
// routing it through a callback measured ~4.7% slower per K chunk on Blackhole.
template <uint32_t B, uint32_t NKV>
struct PieceIter {
    uint32_t c;
    Schedule s;
    uint32_t u0, u1, seg_start, b, h;
    bool done;

    PieceIter(uint32_t core, const Schedule& sched) : c(core), s(sched), seg_start(0), b(0), h(0) {
        u0 = s.U ? s.start(c) : 0;
        u1 = s.U ? s.start(c + 1) : 0;
        done = (u0 >= u1);
    }

    template <typename GetN>
    bool next(GetN get_n, Piece& p) {
        while (!done && b < B) {
            const uint32_t n = get_n(b);
            if (n == 0) {
                ++b;
                h = 0;
                continue;
            }
            const uint32_t cur_h = h;
            const uint32_t cur_b = b;
            const uint32_t seg_end = seg_start + n;
            const bool hit = seg_end > u0 && seg_start < u1;
            const uint32_t this_start = seg_start;
            seg_start = seg_end;
            if (++h == NKV) {
                h = 0;
                ++b;
            }
            if (seg_start >= u1) {
                done = true;
            }
            if (hit) {
                p = Piece{};
                p.b = cur_b;
                p.h = cur_h;
                p.n = n;
                p.j0 = (u0 > this_start ? u0 : this_start) - this_start;
                p.j1 = (u1 < seg_end ? u1 : seg_end) - this_start;
                p.is_child = p.j0 > 0;
                p.is_root = (p.j0 == 0) && (p.j1 < n);
                p.root_core = c;
                if (p.is_child) {
                    p.root_core = s.core_of(this_start);
                    p.child_slot = s.nonempty_between(p.root_core, c);
                }
                if (p.is_root) {
                    p.num_children = s.nonempty_between(c, s.core_of(seg_end - 1) + 1);
                }
                return true;
            }
        }
        return false;
    }
};

}  // namespace lean
