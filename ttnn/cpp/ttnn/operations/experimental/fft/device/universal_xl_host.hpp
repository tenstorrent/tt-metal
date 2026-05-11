// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_host.cpp — XL FFT dispatcher (Option B: host outer twiddle).
//

#pragma once

#include "universal_xl_planner.hpp"
#include "stockham_host.hpp"

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fft_universal_xl {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

namespace detail {

inline uint32_t pick_outer_factor(const XLPlan& p) {
    assert(!p.factors.empty());
    return *std::min_element(p.factors.begin(), p.factors.end());
}

// Outer twiddle table: w[n1*M + k_inner] = exp(-2*pi*i * n1 * k_inner / N).
// Cached per (N, F1) since it's reused on every call with the same shape.
struct OuterTwiddle {
    uint32_t N{0};
    uint32_t F1{0};
    uint32_t M{0};
    std::vector<Complex> w;  // size F1 * M
};

inline std::unordered_map<uint64_t, std::shared_ptr<OuterTwiddle>>&
twiddle_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<OuterTwiddle>> c;
    return c;
}
inline uint64_t twiddle_key(uint32_t N, uint32_t F1) {
    return (static_cast<uint64_t>(N) << 32) | F1;
}

inline std::shared_ptr<OuterTwiddle> get_outer_twiddle(uint32_t N, uint32_t F1) {
    auto key = twiddle_key(N, F1);
    auto& c  = twiddle_cache();
    auto it  = c.find(key);
    if (it != c.end()) return it->second;

    auto tw = std::make_shared<OuterTwiddle>();
    tw->N  = N;
    tw->F1 = F1;
    tw->M  = N / F1;
    tw->w.assign(static_cast<size_t>(F1) * tw->M, Complex{1.0f, 0.0f});

    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        Complex* row = tw->w.data() + static_cast<size_t>(n1) * tw->M;
        for (uint32_t k = 0; k < tw->M; ++k) {
            const double ang = two_pi_over_N
                             * static_cast<double>(n1)
                             * static_cast<double>(k);
            row[k] = Complex{
                static_cast<float>(std::cos(ang)),
                static_cast<float>(std::sin(ang))
            };
        }
    }

    c.emplace(key, tw);
    return tw;
}

inline constexpr uint32_t kXlMaxNFp32_WH = 16u * 1024u * 1024u;   // 16M  (Wormhole)
inline constexpr uint32_t kXlMaxNFp32_BH = 64u * 1024u * 1024u;   // 64M  (Blackhole — ~2x DRAM BW + 2x cores)

inline uint32_t xl_max_n_fp32(const std::shared_ptr<MeshDevice>& md) {
    return (md->arch() == tt::ARCH::BLACKHOLE) ? kXlMaxNFp32_BH
                                               : kXlMaxNFp32_WH;
}

// Back-compat alias used by call sites and error messages — set to the
// most conservative (Wormhole) value so any compile-time use is safe.
inline constexpr uint32_t kXlMaxNFp32 = kXlMaxNFp32_WH;

// Recursive entry: handles pow2 N up to kXlMaxNFp32 by falling through to
// fft_stockham for N <= 1M (k <= 2 in the plan), and applying the
// XL Steps 0-3 above for 1M < N <= 16M.
inline std::vector<Complex> fft_impl(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal,
    const XLPlan&                p)
{
    if (p.single_pass() || p.two_pass()) {
        // N <= 1M: existing fft_stockham handles it directly.
        return fft_stockham::fft(md, signal);
    }

    // K >= 3: outer split.
    const uint32_t F1 = pick_outer_factor(p);
    const uint32_t M  = p.N / F1;
    assert(F1 * M == p.N);
    assert(F1 <= kFactorCap);

    // K >= 4 (N > 1G) needs recursion through fft_universal_xl::fft for
    // the inner length-M sub-problem; not implemented yet.
    TT_FATAL(M <= 1024u * 1024u,
        "fft_universal_xl: N={} inner length M={} exceeds the 1M cap of "
        "fft_stockham. K={} plans (N > 1G) require recursive dispatch — "
        "not yet implemented.",
        p.N, M, p.k());

    const uint32_t n_ceiling = xl_max_n_fp32(md);
    TT_FATAL(p.N <= n_ceiling,
        "fft_universal_xl: N={} above the practical {}M ceiling for this arch "
        "(F1={}, host Step-3 cost ~F1^2 * N ops). The algorithm is correct "
        "here, but the host outer DFT would dominate wall-clock.",
        p.N, n_ceiling >> 20, F1);

    // (dev-time stdout printf removed.)

    std::vector<Complex> T(p.N);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        Complex* tr = T.data() + static_cast<size_t>(n1) * M;
        for (uint32_t n2 = 0; n2 < M; ++n2) {
            tr[n2] = signal[static_cast<size_t>(n2) * F1 + n1];
        }
    }

    // ── Step 1: F1 row-FFTs of length M (sequential) ──────────────────
    // Each row of T is a length-M contiguous buffer ready for fft_stockham.
    std::vector<Complex> Y(p.N);
    std::vector<Complex> row(M);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        const Complex* src = T.data() + static_cast<size_t>(n1) * M;
        std::copy(src, src + M, row.begin());

        std::vector<Complex> y_n1 = fft_stockham::fft(md, row);

        Complex* dst = Y.data() + static_cast<size_t>(n1) * M;
        std::copy(y_n1.begin(), y_n1.end(), dst);
    }

    // ── Step 2: host outer twiddle multiply ───────────────────────────
    auto tw = get_outer_twiddle(p.N, F1);
    for (size_t i = 0; i < Y.size(); ++i) Y[i] *= tw->w[i];

    std::vector<Complex> X(p.N);
    if (F1 == 2u) {
        // Special case: length-2 DFT is just (a + b, a - b).  Tightest
        // possible host loop — one add and one sub per output pair.
        for (uint32_t c = 0; c < M; ++c) {
            const Complex a = Y[/*n1=0*/ 0u * M + c];
            const Complex b = Y[/*n1=1*/ 1u * M + c];
            X[c]            = a + b;          // d = 0
            X[M + c]        = a - b;          // d = 1
        }
    } else {
        // General length-F1 DFT.  Use a tiny per-N cached F1-point twiddle
        // table — same amortisation pattern as the outer twiddle.
        static thread_local std::vector<Complex> sub_tw;
        static thread_local uint32_t             sub_tw_F1 = 0;
        if (sub_tw_F1 != F1) {
            sub_tw.assign(static_cast<size_t>(F1) * F1, Complex{1.0f, 0.0f});
            const double two_pi_over_F1 = -2.0 * M_PI
                                        / static_cast<double>(F1);
            for (uint32_t d = 0; d < F1; ++d) {
                Complex* trow = sub_tw.data() + static_cast<size_t>(d) * F1;
                for (uint32_t a = 0; a < F1; ++a) {
                    const double ang = two_pi_over_F1
                                     * static_cast<double>(d)
                                     * static_cast<double>(a);
                    trow[a] = Complex{
                        static_cast<float>(std::cos(ang)),
                        static_cast<float>(std::sin(ang))
                    };
                }
            }
            sub_tw_F1 = F1;
        }

        std::vector<Complex> v(F1);
        for (uint32_t c = 0; c < M; ++c) {
            for (uint32_t a = 0; a < F1; ++a) {
                v[a] = Y[static_cast<size_t>(a) * M + c];
            }
            for (uint32_t d = 0; d < F1; ++d) {
                const Complex* trow = sub_tw.data()
                                    + static_cast<size_t>(d) * F1;
                Complex acc{0.0f, 0.0f};
                for (uint32_t a = 0; a < F1; ++a) acc += v[a] * trow[a];
                X[static_cast<size_t>(d) * M + c] = acc;
            }
        }
    }
    return X;
}

}  // namespace detail

// ─── Public API ────────────────────────────────────────────────────────────

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2u && "FFT requires N >= 2");
    assert(is_pow2(N) && "fft_universal_xl currently supports pow2 N only");

    const XLPlan p = plan(N);
    return detail::fft_impl(md, signal, p);
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    signal)
{
    std::vector<Complex> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = { signal[i], 0.0f };
    return fft(md, cx);
}

// IFFT via the conjugate trick: ifft(X) = conj(fft(conj(X))) / N.
inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  X)
{
    const uint32_t N = static_cast<uint32_t>(X.size());
    std::vector<Complex> Xc(N);
    for (uint32_t i = 0; i < N; ++i) Xc[i] = std::conj(X[i]);

    std::vector<Complex> y = fft(md, Xc);

    const float inv_N = 1.0f / static_cast<float>(N);
    for (uint32_t i = 0; i < N; ++i) y[i] = std::conj(y[i]) * inv_N;
    return y;
}

}  // namespace fft_universal_xl
