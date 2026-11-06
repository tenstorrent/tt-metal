// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Key-split SparseSDPA: device-side binary reduction tree over the ranks 0..n-1 of one token's core group.
// Same rules as get_tree_reduction_params (sdpa_decode_device_operation.hpp): rank 0 is the root,
// vid = (n-1) - rank, round r: vid receives from vid - 2^r when its low (r+1) bits are all 1, a non-root
// rank sends at round = trailing ones of vid to parent vid + 2^round (an orphan, parent_vid >= n, sends to
// the root). Header-only, no device dependencies, so a host C++ check can include it.

#pragma once

#include <stdint.h>

namespace sparse_sdpa {

constexpr uint32_t MAX_TREE_ROUNDS = 6;  // n <= 64

// The tree word packed into the control message and the gather request.
//   bits 0-1 role (0 idle, 1 non-root worker, 2 root), bits 2-4 send round, bits 5-10 child mask (bit r = this
//   rank receives one child at round r), bits 11-17 parent rank.
namespace tree_word {
constexpr uint32_t ROLE_IDLE = 0;
constexpr uint32_t ROLE_WORKER = 1;
constexpr uint32_t ROLE_ROOT = 2;
constexpr uint32_t pack(uint32_t role, uint32_t send_round, uint32_t child_mask, uint32_t parent) {
    return role | (send_round << 2) | (child_mask << 5) | (parent << 11);
}
constexpr uint32_t role(uint32_t w) { return w & 3u; }
constexpr uint32_t send_round(uint32_t w) { return (w >> 2) & 7u; }
constexpr uint32_t child_mask(uint32_t w) { return (w >> 5) & 63u; }
constexpr uint32_t parent(uint32_t w) { return (w >> 11) & 127u; }
}  // namespace tree_word

constexpr uint32_t ctz32(uint32_t x) {
    uint32_t n = 0;
    while ((x & 1u) == 0u) {
        x >>= 1;
        ++n;
    }
    return n;
}

constexpr uint32_t ceil_log2(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) {
        ++r;
    }
    return r;
}

// Returns the packed tree word for `rank` of an `n`-rank group (n >= 1).
constexpr uint32_t tree_params(uint32_t rank, uint32_t n) {
    if (n <= 1) {
        return tree_word::pack(tree_word::ROLE_ROOT, 0, 0, 0);
    }
    const uint32_t rounds = ceil_log2(n);
    const uint32_t vid = (n - 1) - rank;
    uint32_t child_mask = 0;
    for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t mask = (2u << r) - 1;
        if ((vid & mask) == mask) {
            const uint32_t child_vid = vid - (1u << r);
            if (child_vid < n) {
                child_mask |= 1u << r;
            }
        }
    }
    if (rank != 0) {
        const uint32_t trailing_ones = ctz32(~vid);
        const uint32_t parent_vid = vid + (1u << trailing_ones);
        const uint32_t parent = (parent_vid < n) ? (n - 1) - parent_vid : 0;
        return tree_word::pack(tree_word::ROLE_WORKER, trailing_ones, child_mask, parent);
    }
    // Root: also collects orphans (natural parent_vid >= n) at round t if it has no natural child there.
    for (uint32_t c = 1; c < n; ++c) {
        const uint32_t cv = (n - 1) - c;
        const uint32_t t = ctz32(~cv);
        if (cv + (1u << t) >= n && ((child_mask >> t) & 1u) == 0u) {
            child_mask |= 1u << t;
        }
    }
    return tree_word::pack(tree_word::ROLE_ROOT, 0, child_mask, 0);
}

// Extra words of the key-split control message (reader -> compute) and gather request (reader -> writer).
// Word 0/1 of the control message keep their meaning (ACTIVE_CHUNKS = chunks of THIS core's range,
// VALID_KEYS = nv); the message stays inside the same 16 B page, the request inside the same 32 B page.
namespace ksplit_ctrl {
constexpr uint32_t VALID_LAST = 2;  // valid keys in the last local chunk (k_chunk if the range has no boundary)
constexpr uint32_t TREE = 3;        // packed tree_word
}  // namespace ksplit_ctrl
namespace ksplit_req {
constexpr uint32_t TREE = 5;  // packed tree_word (gather_request words 0-4 are unchanged)
}  // namespace ksplit_req

}  // namespace sparse_sdpa
