// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Group formation for the packed sparse_sdpa_msa reader: the union of up to G tokens' top-k block rows, with
// a "lead" entry every token of the group selected. Plain C++ (no device API), so the host can test it
// (scripts: dev/prefill-sdpa/union_test.cpp).
//
// Semantics (also see sparse_sdpa_msa_packed_reader.cpp):
//  - token 0's ids enter verbatim, in its top-k order (entry e = its c-th id);
//  - a later token's occurrence of block b takes the FIRST entry of b it does not own yet, else appends a new
//    entry, so every token owns exactly the multiset of its own row;
//  - token j joins the group only if some entry is owned by tokens 0..j (the lead); otherwise it is rolled back
//    and starts the next group (g = j);
//  - the first common entry is rotated to index 0.
// Lookups go through a small chained hash on the block id (bucket = id & (kBuckets - 1)) instead of a linear
// scan of the union: ~G*topk probes of ~1 entry each.
#pragma once

#include <stdint.h>

namespace sparse_sdpa_msa_union {

constexpr uint32_t kBuckets = 256;
constexpr uint16_t kNil = 0xFFFF;

// Scratch: head/tail[kBuckets], next[max_entries]. All pointers address caller-owned memory (L1 on device).
template <typename U32P, typename U16P>
struct Union {
    U32P id;    // [max_entries] block id of each entry
    U32P mask;  // [max_entries] bit j = token j of the group owns the entry
    U16P head;  // [kBuckets] first entry of the bucket's chain (kNil = empty)
    U16P tail;  // [kBuckets] last entry of the bucket's chain
    U16P next;  // [max_entries] next entry in the chain
    uint32_t n = 0;

    void reset() {
        for (uint32_t b = 0; b < kBuckets; ++b) {
            head[b] = kNil;
        }
        n = 0;
    }
    void link(uint32_t e) {
        const uint32_t b = id[e] & (kBuckets - 1);
        next[e] = kNil;
        if (head[b] == kNil) {
            head[b] = static_cast<uint16_t>(e);
        } else {
            next[tail[b]] = static_cast<uint16_t>(e);
        }
        tail[b] = static_cast<uint16_t>(e);
    }
    uint32_t append(uint32_t blk, uint32_t m) {
        id[n] = blk;
        mask[n] = m;
        link(n);
        return n++;
    }
    // First entry of `blk` whose mask lacks `bit`, or n (none).
    uint32_t find_free(uint32_t blk, uint32_t bit) const {
        for (uint32_t e = head[blk & (kBuckets - 1)]; e != kNil; e = next[e]) {
            if (id[e] == blk && (mask[e] & bit) == 0) {
                return e;
            }
        }
        return n;
    }
    // First entry of `blk` owned by `bit`, or n (none).
    uint32_t find_owned(uint32_t blk, uint32_t bit) const {
        for (uint32_t e = head[blk & (kBuckets - 1)]; e != kNil; e = next[e]) {
            if (id[e] == blk && (mask[e] & bit) != 0) {
                return e;
            }
        }
        return n;
    }
    void rebuild_chains() {
        for (uint32_t b = 0; b < kBuckets; ++b) {
            head[b] = kNil;
        }
        for (uint32_t e = 0; e < n; ++e) {
            link(e);
        }
    }
};

// rows: token j's ids at rows[j * topk + c], c < nv[j]. Returns g (>= 1); u holds the union (lead at entry 0,
// chains rebuilt for the final order so find_owned works on the rotated indices).
template <typename RowP, typename U32P, typename U16P>
uint32_t build_group(Union<U32P, U16P>& u, RowP rows, const uint32_t* nv, uint32_t g_max, uint32_t topk) {
    u.reset();
    const uint32_t nv0 = nv[0];
    for (uint32_t c = 0; c < nv0; ++c) {
        u.append(rows[c], 1u);
    }
    uint32_t g = 1;
    for (uint32_t j = 1; j < g_max; ++j) {
        const uint32_t bit = 1u << j;
        const uint32_t n_before = u.n;
        for (uint32_t c = 0; c < nv[j]; ++c) {
            const uint32_t b = rows[j * topk + c];
            const uint32_t e = u.find_free(b, bit);
            if (e == u.n) {
                u.append(b, bit);
            } else {
                u.mask[e] |= bit;
            }
        }
        // The lead must be one of token 0's entries [0, nv0).
        const uint32_t all = (bit << 1) - 1;
        bool has_lead = false;
        for (uint32_t e = 0; e < nv0 && !has_lead; ++e) {
            has_lead = (u.mask[e] & all) == all;
        }
        if (!has_lead) {
            u.n = n_before;
            for (uint32_t e = 0; e < n_before; ++e) {
                u.mask[e] &= ~bit;
            }
            break;
        }
        g = j + 1;
    }
    const uint32_t all = (1u << g) - 1;
    uint32_t lead = 0;
    while ((u.mask[lead] & all) != all) {
        ++lead;
    }
    if (lead != 0) {
        const uint32_t lid = u.id[lead], lmask = u.mask[lead];
        for (uint32_t e = lead; e > 0; --e) {
            u.id[e] = u.id[e - 1];
            u.mask[e] = u.mask[e - 1];
        }
        u.id[0] = lid;
        u.mask[0] = lmask;
    }
    u.rebuild_chains();
    return g;
}

}  // namespace sparse_sdpa_msa_union
