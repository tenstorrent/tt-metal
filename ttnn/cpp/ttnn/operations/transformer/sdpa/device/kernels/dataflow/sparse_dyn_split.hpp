// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace sparse_dyn_split {

constexpr uint32_t kMaxUnits = 64;

struct Entry {
    bool ok = false;
    uint32_t unit = 0;
    uint32_t q_start = 0;
    uint32_t q_count = 0;
    uint32_t unit_core = 0;
    uint32_t unit_cores = 0;
    uint32_t first = 0;
};

inline uint32_t unit_count(uint32_t n, uint32_t c, uint32_t j) {
    return static_cast<uint32_t>(uint64_t(j + 1) * n / c - uint64_t(j) * n / c);
}

inline Entry entry(
    const uint32_t* w,
    uint16_t* c,
    uint32_t units,
    uint32_t n,
    uint32_t cores,
    uint32_t max_q,
    uint32_t min_q,
    uint32_t core) {
    Entry e;
    if (units == 0 || units > kMaxUnits || n == 0 || cores < units) {
        return e;
    }
    const uint32_t mq = max_q > 0 ? max_q : 1;
    const uint32_t mn = min_q > 0 ? min_q : 1;
    const uint32_t c0 = (n + mq - 1) / mq > 1 ? (n + mq - 1) / mq : 1;
    if (c0 * units > cores || c0 * mn > n) {
        return e;
    }
    for (uint32_t u = 0; u < units; ++u) {
        c[u] = static_cast<uint16_t>(c0);
    }
    for (uint32_t spare = cores - c0 * units; spare > 0; --spare) {
        uint32_t best = units;
        uint64_t best_cost = 0;
        bool have = false;
        for (uint32_t u = 0; u < units; ++u) {
            if ((uint32_t(c[u]) + 1) * mn > n) {
                continue;
            }
            const uint64_t cost = static_cast<uint64_t>((n + c[u] - 1) / c[u]) * w[u];
            if (!have || cost > best_cost) {
                have = true;
                best_cost = cost;
                best = u;
            }
        }
        if (best == units) {
            return e;
        }
        ++c[best];
    }
    uint32_t first = 0;
    for (uint32_t u = 0; u < units; ++u) {
        if (core < first + c[u]) {
            const uint32_t j = core - first;
            e.ok = true;
            e.unit = u;
            e.unit_core = j;
            e.unit_cores = c[u];
            e.first = first;
            e.q_start = u * n + static_cast<uint32_t>(uint64_t(j) * n / c[u]);
            e.q_count = unit_count(n, c[u], j);
            return e;
        }
        first += c[u];
    }
    return e;
}

}  // namespace sparse_dyn_split
