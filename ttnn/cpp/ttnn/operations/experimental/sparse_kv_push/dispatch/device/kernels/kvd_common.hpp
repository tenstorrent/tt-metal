// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch: shared constants, the wire and queue entry layout, and the tree routing (sparse_dispatch_2d's
// span rule, as tests/nightly/blackhole/sdpa/kvd2d_routing.py ports it). Plain
// constexpr / inline only: the host includes it too.
#pragma once

#include <cstdint>

namespace kvd {

// Debug statistics (the host adds the KVD_STATS define when the KVD_STATS env var is set): profiler zones, a wall
// clock for per-core cycle counters, and one DPRINT a core. Without it they compile away.
#ifdef KVD_STATS
#define KVD_ZONE(name) DeviceZoneScopedN(name)
inline uint32_t stats_clock() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
#else
#define KVD_ZONE(name)
inline uint32_t stats_clock() { return 0; }
#endif

// ---- Port cores: one per (lane, direction), at a fixed index so the same logical core faces the same way everywhere.
// Directions: 0 X+ (column + 1), 1 X- , 2 Y+ (row + 1), 3 Y-. A packet sent by port d moves in d and lands on the
// neighbour's port d ^ 1 (the one facing back). A lane is an independent instance of the protocol: every hop of a
// row's tree stays on the row's lane, and lane l uses fabric link l % links (its link's lanes share it through a mux).
constexpr uint32_t kDirections = 4;
constexpr uint32_t kMaxLinks = 2;
constexpr uint32_t kMaxLanes = 8;
constexpr uint32_t kMaxPorts = kDirections * kMaxLanes;
constexpr uint32_t port_index(uint32_t lane, uint32_t dir) { return lane * kDirections + dir; }

// ---- Entries (wire, own lists and queues): two words.
// word 0: row (22 bits) | hops left << 22 (3 bits) | generation << 25 (4 bits) | order << 29 | END << 31
// word 1: destination mask (bit i: chip i, row-major)
constexpr uint32_t kRowBits = 22, kHopsShift = 22, kHopsBits = 3, kGenShift = 25, kGenBits = 4, kOrderShift = 29;
constexpr uint32_t kEndBit = 1u << 31;
constexpr uint32_t kMaxRows = 1u << kRowBits;
constexpr uint32_t kEntryBytes = 8;
constexpr uint32_t entry_word0(uint32_t row, uint32_t hops, uint32_t gen, uint32_t order) {
    return row | (hops << kHopsShift) | (gen << kGenShift) | (order << kOrderShift);
}
constexpr uint32_t end_word0(uint32_t gen) { return kEndBit | (gen << kGenShift); }
constexpr uint32_t entry_row(uint32_t w0) { return w0 & ((1u << kRowBits) - 1); }
constexpr uint32_t entry_hops(uint32_t w0) { return (w0 >> kHopsShift) & ((1u << kHopsBits) - 1); }
constexpr uint32_t entry_gen(uint32_t w0) { return (w0 >> kGenShift) & ((1u << kGenBits) - 1); }
constexpr uint32_t entry_order(uint32_t w0) { return (w0 >> kOrderShift) & 1; }

// ---- A landing slot = one packet: header, then up to B entries, then the rows in entry order (END has no row).
constexpr uint32_t kSlotHeaderBytes = 16;  // word 0: entries, word 1: rows
// DRAM <-> L1 transfers need 64 B alignment on Blackhole: rows start 64 B aligned in a slot, slots are 64 B multiples,
// and every L1 buffer a kernel moves rows or entries through starts at align64(its CB's base) (CBs get 64 B of slack).
constexpr uint32_t align64(uint32_t x) { return (x + 63) & ~63u; }
constexpr uint32_t slot_rows_offset(uint32_t rows_per_packet) {
    return align64(kSlotHeaderBytes + rows_per_packet * kEntryBytes);
}
constexpr uint32_t slot_bytes(uint32_t rows_per_packet, uint32_t row_bytes) {
    return align64(slot_rows_offset(rows_per_packet) + rows_per_packet * row_bytes);
}

// A sender builds up to this many packets a round (one read barrier, then all sent).
constexpr uint32_t kRoundPackets = 4;

// ---- Scratch (DRAM, pages of kScratchPageBytes): own lists, then onward queues.
constexpr uint32_t kScratchPageBytes = 4096;
constexpr uint32_t kEntriesPerPage = kScratchPageBytes / kEntryBytes;
// Input streams of a port: 0 continue (packets moving its direction), 1 / 2 turns from packets moving + / - along the
// other axis.
constexpr uint32_t kStreams = 3;

// ---- Row hash (Knuth): bit 30 every ring tie, bit 31 the order tie, bit 29 the link.
constexpr uint32_t kHash = 2654435761u;
constexpr uint32_t hbit(uint32_t row, uint32_t bit) { return ((row * kHash) >> bit) & 1; }
constexpr uint32_t kTieBit = 30, kOrderBit = 31, kLinkBit = 29;
// A row's lane: hash bits 21..28 modulo the lane count (bits 29..31 pick the tree, so lanes stay independent of it).
constexpr uint32_t lane_of(uint32_t row, uint32_t lanes) { return (((row * kHash) >> 21) & 0xFF) % lanes; }

// ---- Spans. A span byte: hops forward | hops backward << 4.
constexpr uint32_t span_fwd(uint32_t s) { return s & 0xF; }
constexpr uint32_t span_bwd(uint32_t s) { return (s >> 4) & 0xF; }

// The span from pos covering the set positions of line_mask on an axis of n chips. A ring uses `table` (2^n x 2 bytes,
// indexed by the mask rotated to pos, then the tie bit; built on the host by the same rule as sparse_dispatch_2d's
// span_entry); a line's span is forced.
template <uint32_t n, bool wrap>
inline uint32_t span_of(const uint8_t* table, uint32_t pos, uint32_t line_mask, uint32_t tie) {
    uint32_t m = line_mask & ~(1u << pos) & ((1u << n) - 1);
    if (m == 0) {
        return 0;
    }
    if constexpr (wrap) {
        const uint32_t rot = ((m >> pos) | (m << (n - pos))) & ((1u << n) - 1);
        return table[rot * 2 + tie];
    } else {
        const uint32_t hi = 31 - __builtin_clz(m), lo = __builtin_ctz(m);
        return (hi > pos ? hi - pos : 0) | ((lo < pos ? pos - lo : 0) << 4);
    }
}

// The mesh's routing: R rows x C columns, row-major chips, each axis a ring or a line.
template <uint32_t R, uint32_t C, bool WX, bool WY>
struct Routing {
    const uint8_t* xring;
    const uint8_t* yring;

    static constexpr uint32_t cols_of(uint32_t m) {
        uint32_t cm = 0;
        for (uint32_t r = 0; r < R; ++r) {
            cm |= (m >> (r * C)) & ((1u << C) - 1);
        }
        return cm;
    }
    static constexpr uint32_t rows_of(uint32_t m) {
        uint32_t rm = 0;
        for (uint32_t r = 0; r < R; ++r) {
            rm |= (((m >> (r * C)) & ((1u << C) - 1)) != 0 ? 1u : 0u) << r;
        }
        return rm;
    }
    static constexpr uint32_t rows_in_col(uint32_t m, uint32_t c) {
        uint32_t rm = 0;
        for (uint32_t r = 0; r < R; ++r) {
            rm |= ((m >> (r * C + c)) & 1) << r;
        }
        return rm;
    }
    static constexpr uint32_t cols_in_row(uint32_t m, uint32_t r) { return (m >> (r * C)) & ((1u << C) - 1); }

    uint32_t xspan(uint32_t c, uint32_t cm, uint32_t tie) const { return span_of<C, WX>(xring, c, cm, tie); }
    uint32_t yspan(uint32_t r, uint32_t rm, uint32_t tie) const { return span_of<R, WY>(yring, r, rm, tie); }

    // first leg along axis_x (else Y) from (r, c): over every column (row) holding a destination
    uint32_t first_leg(bool axis_x, uint32_t r, uint32_t c, uint32_t m, uint32_t tie) const {
        return axis_x ? xspan(c, cols_of(m), tie) : yspan(r, rows_of(m), tie);
    }
    // turn at (r, c) into_x (else Y): over the destinations in this chip's row (column)
    uint32_t turn(bool into_x, uint32_t r, uint32_t c, uint32_t m, uint32_t tie) const {
        return into_x ? xspan(c, cols_in_row(m, r), tie) : yspan(r, rows_in_col(m, c), tie);
    }
    // edges of the row's tree in an order (0: X first)
    uint32_t edges(uint32_t r0, uint32_t c0, uint32_t m, uint32_t order, uint32_t tie) const {
        const uint32_t f = first_leg(order == 0, r0, c0, m, tie);
        uint32_t n = span_fwd(f) + span_bwd(f);
        const int32_t fw = static_cast<int32_t>(span_fwd(f)), bw = static_cast<int32_t>(span_bwd(f));
        for (int32_t k = -bw; k <= fw; ++k) {
            uint32_t t;
            if (order == 0) {
                const uint32_t c = static_cast<uint32_t>(
                    (static_cast<int32_t>(c0) + k + 4 * static_cast<int32_t>(C)) % static_cast<int32_t>(C));
                t = turn(false, r0, c, m, tie);
            } else {
                const uint32_t r = static_cast<uint32_t>(
                    (static_cast<int32_t>(r0) + k + 4 * static_cast<int32_t>(R)) % static_cast<int32_t>(R));
                t = turn(true, r, c0, m, tie);
            }
            n += span_fwd(t) + span_bwd(t);
        }
        return n;
    }
    uint32_t order_of(uint32_t r0, uint32_t c0, uint32_t m, uint32_t row) const {
        const uint32_t tie = hbit(row, kTieBit);
        const uint32_t e0 = edges(r0, c0, m, 0, tie), e1 = edges(r0, c0, m, 1, tie);
        return e0 < e1 ? 0 : e1 < e0 ? 1 : hbit(row, kOrderBit);
    }
};

// A row's destination mask split by line, computed once per row: everything the order choice and the owner's first
// hops need, as table lookups (the planner's hot path).
template <uint32_t R, uint32_t C, bool WX, bool WY>
struct RowTree {
    uint32_t row_bits[R];  // destinations in mesh row r, as a column mask
    uint32_t col_bits[C];  // destinations in mesh column c, as a row mask
    uint32_t rows_mask = 0, cols_mask = 0;

    // bits 0, 4, ..., 28 of x -> bits 0..7
    static uint32_t gather_nibble_lsbs(uint32_t x) {
        x &= 0x11111111u;
        x = (x | (x >> 3)) & 0x03030303u;
        x = (x | (x >> 6)) & 0x000F000Fu;
        return (x | (x >> 12)) & 0xFFu;
    }
    RowTree(uint32_t m) {
        if constexpr (R == 8 && C == 4) {  // (the Galaxy mesh: 8 nibbles, no loops)
            for (uint32_t r = 0; r < R; ++r) {
                row_bits[r] = (m >> (r * C)) & 0xFu;
            }
            uint32_t o = m | (m >> 16);
            o |= o >> 8;
            cols_mask = (o | (o >> 4)) & 0xFu;
            uint32_t y = m | (m >> 1);
            rows_mask = gather_nibble_lsbs(y | (y >> 2));
            for (uint32_t c = 0; c < C; ++c) {
                col_bits[c] = gather_nibble_lsbs(m >> c);
            }
            return;
        }
        for (uint32_t c = 0; c < C; ++c) {
            col_bits[c] = 0;
        }
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t b = (m >> (r * C)) & ((1u << C) - 1);
            row_bits[r] = b;
            cols_mask |= b;
            rows_mask |= (b != 0 ? 1u : 0u) << r;
            for (uint32_t c = 0; c < C; ++c) {
                col_bits[c] |= ((b >> c) & 1) << r;
            }
        }
    }
    // edges of the tree in an order (0: X first) from (r0, c0)
    uint32_t edges(const Routing<R, C, WX, WY>& rt, uint32_t r0, uint32_t c0, uint32_t order, uint32_t tie) const {
        if (order == 0) {
            const uint32_t f = rt.xspan(c0, cols_mask, tie);
            uint32_t n = span_fwd(f) + span_bwd(f);
            for (int32_t k = -static_cast<int32_t>(span_bwd(f)); k <= static_cast<int32_t>(span_fwd(f)); ++k) {
                const uint32_t c = static_cast<uint32_t>((static_cast<int32_t>(c0 + C) + k) % static_cast<int32_t>(C));
                const uint32_t t = rt.yspan(r0, col_bits[c], tie);
                n += span_fwd(t) + span_bwd(t);
            }
            return n;
        }
        const uint32_t f = rt.yspan(r0, rows_mask, tie);
        uint32_t n = span_fwd(f) + span_bwd(f);
        for (int32_t k = -static_cast<int32_t>(span_bwd(f)); k <= static_cast<int32_t>(span_fwd(f)); ++k) {
            const uint32_t r = static_cast<uint32_t>((static_cast<int32_t>(r0 + R) + k) % static_cast<int32_t>(R));
            const uint32_t t = rt.xspan(c0, row_bits[r], tie);
            n += span_fwd(t) + span_bwd(t);
        }
        return n;
    }
    uint32_t order(const Routing<R, C, WX, WY>& rt, uint32_t r0, uint32_t c0, uint32_t row) const {
        const uint32_t tie = hbit(row, kTieBit);
        const uint32_t e0 = edges(rt, r0, c0, 0, tie), e1 = edges(rt, r0, c0, 1, tie);
        return e0 < e1 ? 0 : e1 < e0 ? 1 : hbit(row, kOrderBit);
    }
    // the owner's hops in direction d (0 X+, 1 X-, 2 Y+, 3 Y-): its first leg, or its own turn
    uint32_t own_hops(
        const Routing<R, C, WX, WY>& rt, uint32_t r0, uint32_t c0, uint32_t order, uint32_t d, uint32_t tie) const {
        const bool axis_x = d < 2;
        uint32_t sp;
        if (axis_x) {
            sp = order == 0 ? rt.xspan(c0, cols_mask, tie) : rt.xspan(c0, row_bits[r0], tie);
        } else {
            sp = order == 1 ? rt.yspan(r0, rows_mask, tie) : rt.yspan(r0, col_bits[c0], tie);
        }
        return (d & 1) == 0 ? span_fwd(sp) : span_bwd(sp);
    }
};

}  // namespace kvd
