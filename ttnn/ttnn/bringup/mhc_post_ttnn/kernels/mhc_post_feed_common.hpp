// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed (deepseek_v41_flash_d_p P.36): what a port carries, shared by its reader and its sender.
// Port (side d, link l) carries to target t: the units u in [u_lo, u_hi) with (u - u_lo) mod L == l, of each the
// column tiles k in [t cpt, (t + 1) cpt) (unit u: block b = u / RT, row tile r = u mod RT, columns [b B, b B + valid)).
// Packets: up to P tiles a packet; the stream's last tile goes alone (its packet carries the arrival increment), so
// a packet's size is P while more than P tiles are left, then left - 1, then 1.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/fabric_common.h"
#include "mhc_post_feed_layout.hpp"
#include "mhc_post_stats_common.hpp"

namespace mhc_feed {

// The feed table (mhc_post_feed_layout.hpp) at common arg FT0, and this chip's place in it: the fabric routing table
// in L1 names the chip (my_device_id), the table maps it to its mesh position and group position me.
template <uint32_t FT0>
struct Table {
    static uint32_t w(uint32_t i) { return get_common_arg_val<uint32_t>(FT0 + i); }
    static uint32_t byte(uint32_t base, uint32_t k) { return (w(base + k / 4) >> (8 * (k % 4))) & 0xFFu; }
    uint32_t G, axis, C, L, mesh, pos, me;
    Table() {
        using namespace mhc_feed_layout;
        G = w(FT_G);
        axis = w(FT_AXIS);
        C = w(FT_C);
        L = w(FT_L);
        mesh = w(FT_MESH);
        const uint32_t dev =
            reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE)->my_device_id;
        pos = byte(FT_POS_OF_CHIP, dev);
        me = axis == 1 ? pos % C : pos / C;
    }
    // the side of my port toward group position b from a (0 / 1)
    uint32_t side(uint32_t a, uint32_t b) const {
        const uint32_t k = a * G + b;
        return (w(mhc_feed_layout::FT_SIDE + k / 32) >> (k % 32)) & 1u;
    }
    uint32_t chip_of(uint32_t t) const {  // the fabric chip id of my group's position t
        const uint32_t p = axis == 1 ? (pos / C) * C + t : t * C + pos % C;
        return byte(mhc_feed_layout::FT_CHIP_OF_POS, p);
    }
    uint32_t port_xy(uint32_t slot) const { return w(mhc_feed_layout::FT_PORTS + slot); }
    uint32_t u_lo(uint32_t t) const { return w(mhc_feed_layout::FT_ULO + t); }
    uint32_t u_hi(uint32_t t) const { return w(mhc_feed_layout::FT_UHI + t); }
    uint32_t dir(uint32_t d) const { return w(mhc_feed_layout::FT_DIR + d); }
};

template <typename F>
inline void for_each_tile(
    uint32_t t,
    uint32_t u_lo,
    uint32_t u_hi,
    uint32_t l,
    uint32_t links,
    uint32_t rt,
    uint32_t bw,
    uint32_t ct,
    uint32_t cpt,
    F&& fn) {
    const uint32_t k_lo = t * cpt, k_hi = k_lo + cpt;
    for (uint32_t u = u_lo + l; u < u_hi; u += links) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        const uint32_t c0 = U.col0 < k_lo ? k_lo - U.col0 : 0;
        const uint32_t c1 = U.col0 + U.valid > k_hi ? k_hi - U.col0 : U.valid;
        for (uint32_t c = c0; c < c1; ++c) {
            fn(U.r, U.col0 + c);
        }
    }
}

inline uint32_t count_tiles(
    uint32_t t,
    uint32_t u_lo,
    uint32_t u_hi,
    uint32_t l,
    uint32_t links,
    uint32_t rt,
    uint32_t bw,
    uint32_t ct,
    uint32_t cpt) {
    uint32_t n = 0;
    for_each_tile(t, u_lo, u_hi, l, links, rt, bw, ct, cpt, [&](uint32_t, uint32_t) { ++n; });
    return n;
}

// the size of the next packet with `left` tiles of the stream left (left >= 1)
inline uint32_t packet_size(uint32_t left, uint32_t p) {
    if (left > p) {
        return p;
    }
    return left > 1 ? left - 1 : 1;
}

}  // namespace mhc_feed
