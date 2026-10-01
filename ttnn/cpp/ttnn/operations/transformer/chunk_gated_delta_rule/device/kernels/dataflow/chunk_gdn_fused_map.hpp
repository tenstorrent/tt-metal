// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused chunk_gdn producer map: which producer computes chunk c of head h, and each producer's item list.
// Included by the host factory and the three fused dataflow kernels so both sides evaluate one formula.
//
// Producers 0 .. BH*NPH-1 are HOME producers: producer h*NPH + j is head h's j-th. Producers BH*NPH ..
// BH*NPH+NX-1 are the pool's EXTRAS, which take the share num/den of every head's chunks. The extra items
// e = 0, 1, ... are spread evenly over the BH*NC (chunk, head) slots: item e is head e % BH at chunk
// ceil((e+1)*den/(num*BH)) - 1, so chunk c of head h is an extra chunk iff
// floor(((c+1)*num*BH + (BH-1-h)*den) / (den*BH)) > floor((c*num*BH + (BH-1-h)*den) / (den*BH)) — the extra
// chunks of consecutive heads are phased by den/(num*BH) chunks, so an extra's successive items step through
// increasing chunks instead of the same chunk of BH/NX heads. Extra x serves the items x, x+NX, ...; home
// producer j of head h serves that head's non-extra chunks of ranks r = j, j+NPH, ... in chunk order. Every
// producer's chunks are therefore non-decreasing. The per-head form (NP producers per head, producer j owns
// c = j, j+NP, ...) is NX = 0.

#pragma once

#include <cstdint>

struct GdnFusedMap {
    uint32_t BH;   // heads
    uint32_t NC;   // chunks per head
    uint32_t NPH;  // home producers per head, >= 1
    uint32_t NX;   // extra producers
    uint32_t num;  // extras' share of each head's chunks = num / den; num <= den, den >= 1 (ignored when NX == 0)
    uint32_t den;
};

struct GdnFusedItem {
    uint32_t h;
    uint32_t c;
};

// Extra chunks of head h in [0, c).
inline uint32_t gdn_fused_extras_before(const GdnFusedMap& m, uint32_t h, uint32_t c) {
    return m.NX != 0 ? (c * m.num * m.BH + (m.BH - 1 - h) * m.den) / (m.den * m.BH) : 0u;
}

inline bool gdn_fused_is_extra(const GdnFusedMap& m, uint32_t h, uint32_t c) {
    return gdn_fused_extras_before(m, h, c + 1) != gdn_fused_extras_before(m, h, c);
}

inline uint32_t gdn_fused_n_home_chunks(const GdnFusedMap& m, uint32_t h) {
    return m.NC - gdn_fused_extras_before(m, h, m.NC);
}

// Extra items over all heads.
inline uint32_t gdn_fused_n_extra_items(const GdnFusedMap& m) { return m.NX != 0 ? (m.NC * m.num * m.BH) / m.den : 0u; }

// Chunk of extra item e (head e % BH). Needs num >= 1.
inline uint32_t gdn_fused_extra_chunk(const GdnFusedMap& m, uint32_t e) {
    return ((e + 1) * m.den + m.num * m.BH - 1) / (m.num * m.BH) - 1;
}

// Chunk of home rank r of head h: the smallest c with r + 1 non-extra chunks in [0, c].
inline uint32_t gdn_fused_home_chunk(const GdnFusedMap& m, uint32_t h, uint32_t r) {
    return (m.NX != 0 && m.num < m.den) ? ((r * m.BH + m.BH - 1 - h) * m.den) / (m.BH * (m.den - m.num)) : r;
}

// Producer of chunk c of head h.
inline uint32_t gdn_fused_owner(const GdnFusedMap& m, uint32_t h, uint32_t c) {
    const uint32_t xb = gdn_fused_extras_before(m, h, c);
    if (gdn_fused_is_extra(m, h, c)) {
        return m.BH * m.NPH + (xb * m.BH + h) % m.NX;
    }
    return h * m.NPH + (c - xb) % m.NPH;
}

inline uint32_t gdn_fused_item_count(const GdnFusedMap& m, uint32_t p) {
    if (p < m.BH * m.NPH) {
        const uint32_t j = p % m.NPH;
        const uint32_t nh = gdn_fused_n_home_chunks(m, p / m.NPH);
        return nh > j ? (nh - j + m.NPH - 1) / m.NPH : 0u;
    }
    const uint32_t x = p - m.BH * m.NPH;
    const uint32_t ne = gdn_fused_n_extra_items(m);
    return ne > x ? (ne - x + m.NX - 1) / m.NX : 0u;
}

// Producer p's n-th item, n < gdn_fused_item_count(m, p).
inline GdnFusedItem gdn_fused_item(const GdnFusedMap& m, uint32_t p, uint32_t n) {
    if (p < m.BH * m.NPH) {
        const uint32_t h = p / m.NPH;
        return {h, gdn_fused_home_chunk(m, h, p % m.NPH + n * m.NPH)};
    }
    const uint32_t e = p - m.BH * m.NPH + n * m.NX;
    return {e % m.BH, gdn_fused_extra_chunk(m, e)};
}
