// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_planner.hpp — Pure host K-level factorisation planner.
//

#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace fft_universal_xl {

// Hard limit set by the existing fft_stockham::batch_fft kernel:
// each sub-FFT must fit in one Tensix tile = 32 * 32 = 1024 elements.
inline constexpr uint32_t kFactorCap = 1024u;

inline constexpr bool is_pow2(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

inline constexpr uint32_t log2u(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) ++r;
    return r;
}

// ─── Plan struct ────────────────────────────────────────────────────────────

struct XLPlan {
    uint32_t              N{0};
    std::vector<uint32_t> factors;   // [F1, F2, ..., Fk], prod == N

    // Convenience accessors.
    uint32_t k() const { return static_cast<uint32_t>(factors.size()); }

    bool single_pass() const { return k() == 1u; }
    bool two_pass()    const { return k() == 2u; }
    bool deep()        const { return k() >= 3u; }
};

inline XLPlan plan(uint32_t N) {
    assert(is_pow2(N) && "fft_universal_xl::plan currently requires pow2 N");
    assert(N >= 2u    && "fft_universal_xl::plan requires N >= 2");

    XLPlan p;
    p.N = N;

    if (N <= kFactorCap) {
        // Single-pass: existing kernel handles directly.
        p.factors = { N };
        return p;
    }

    const uint32_t log2N      = log2u(N);
    const uint32_t log2_cap   = log2u(kFactorCap);
    const uint32_t k          = (log2N + log2_cap - 1u) / log2_cap;   // ceil
    p.factors.reserve(k);

    uint32_t remaining_log2 = log2N;
    for (uint32_t i = 0; i + 1u < k; ++i) {
        p.factors.push_back(kFactorCap);
        remaining_log2 -= log2_cap;
    }
    p.factors.push_back(1u << remaining_log2);

    // Sanity: factor product == N and each factor <= cap.
    // [[maybe_unused]] because asserts are stripped in Release (-DNDEBUG)
    // and the variable is only consumed by them.
    [[maybe_unused]] uint64_t prod = 1ull;
    for (uint32_t f : p.factors) {
        assert(f <= kFactorCap && "factor exceeds tile cap");
        assert(is_pow2(f)      && "factor must be pow2");
        prod *= f;
    }
    assert(prod == static_cast<uint64_t>(N));

    return p;
}

// ─── Helpers used by the dispatcher (Phase 1 / Phase 2) ─────────────────────

// Largest power-of-two factor of N that is <= kFactorCap.  Used by Phase 2
// when re-decomposing tail factors mid-recursion.  Returns 1 if no such
// factor exists (impossible for pow2 N >= 2).
inline uint32_t largest_pow2_factor_le_cap(uint32_t N) {
    uint32_t f = (N <= kFactorCap) ? N : kFactorCap;
    while (f > 1u && (N % f) != 0u) f >>= 1;
    return f;
}

inline std::pair<uint32_t, uint32_t> outer_split(const XLPlan& p) {
    assert(!p.factors.empty());
    if (p.factors.size() == 1u) {
        return { p.factors.front(), 1u };
    }
    const uint32_t F1 = p.factors.front();
    return { F1, p.N / F1 };
}

// Same idea but stripping the OUTER (first) factor, returning the plan
// for the inner length-M sub-problem.  Phase 1 dispatcher uses this to
// recurse via fft_stockham::fft when the inner length is <= 1M.
inline XLPlan strip_outer(const XLPlan& p) {
    assert(!p.factors.empty());
    XLPlan q;
    q.factors.assign(p.factors.begin() + 1, p.factors.end());
    uint64_t prod = 1ull;
    for (uint32_t f : q.factors) prod *= f;
    q.N = static_cast<uint32_t>(prod);
    return q;
}

}  // namespace fft_universal_xl
