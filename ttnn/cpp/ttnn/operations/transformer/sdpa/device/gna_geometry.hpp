// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// GNA fused kernel host geometry -- C++ port of models/tt_dit/layers/gna_gather.py::mask_table.
// Deterministic function of (grid, block, kernel), so the program factory derives it (no new op inputs).
// box_idx + table[mask_id] reproduces exact neighborhood (validated in Python: torch 100%, device op 99.975%).
// Region grouping (K/V reuse across adjacent blocks) is a LATER iteration; here region = 1 block (per-block box).

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace ttnn::operations::transformer::gna {

inline uint32_t nbr_start(uint32_t q, uint32_t L, uint32_t ker) {
    if (ker > L) {
        ker = L;
    }
    const uint32_t half = ker / 2;
    const uint32_t last = L - ker;
    uint32_t s = q < half ? 0u : q - half;
    return s > last ? last : s;
}

inline uint32_t clamp_origin(uint32_t bi, uint32_t b, uint32_t half, uint32_t ext, uint32_t L) {
    int32_t o = static_cast<int32_t>(bi * b) - static_cast<int32_t>(half);
    if (o < 0) {
        o = 0;
    }
    if (o > static_cast<int32_t>(L - ext)) {
        o = static_cast<int32_t>(L - ext);
    }
    return static_cast<uint32_t>(o);
}

struct GnaGeometry {
    std::vector<uint32_t> box_idx;    // nb * box_vol  (flat t_inner seq indices per box cell, per block)
    std::vector<uint8_t> mask_table;  // n_distinct * vol * box_vol  (1 = in-window/live)
    std::vector<uint32_t> mask_id;    // nb  -> row of mask_table
    uint32_t nb = 0, vol = 0, box_vol = 0, n_distinct = 0;
    uint32_t ext_t = 0, ext_h = 0, ext_w = 0;
};

// grid = (T,H,W) op-order (t_inner: T innermost in the flat index ((w*H+h)*T+t)); block/kernel same order.
inline GnaGeometry compute_gna_geometry(
    uint32_t T, uint32_t H, uint32_t W, uint32_t bt, uint32_t bh, uint32_t bw, uint32_t kt, uint32_t kh, uint32_t kw) {
    GnaGeometry g;
    const uint32_t Tb = T / bt, Hb = H / bh, Wb = W / bw;
    const uint32_t ker_t = kt > T ? T : kt, ker_h = kh > H ? H : kh, ker_w = kw > W ? W : kw;
    g.ext_t = bt + ker_t - 1;
    g.ext_h = bh + ker_h - 1;
    g.ext_w = bw + ker_w - 1;
    const uint32_t ht = ker_t / 2, hh = ker_h / 2, hw = ker_w / 2;
    g.vol = bt * bh * bw;
    g.box_vol = g.ext_t * g.ext_h * g.ext_w;
    g.nb = Tb * Hb * Wb;
    g.box_idx.resize(static_cast<size_t>(g.nb) * g.box_vol);
    g.mask_id.resize(g.nb);

    std::map<std::array<int32_t, 3>, std::pair<uint32_t, uint32_t>> combos;  // (offsets) -> (row, rep block)
    for (uint32_t b = 0; b < g.nb; ++b) {
        const uint32_t bti = b / (Hb * Wb), bhi = (b / Wb) % Hb, bwi = b % Wb;
        const uint32_t t0 = clamp_origin(bti, bt, ht, g.ext_t, T);
        const uint32_t h0 = clamp_origin(bhi, bh, hh, g.ext_h, H);
        const uint32_t w0 = clamp_origin(bwi, bw, hw, g.ext_w, W);
        uint32_t c = 0;
        for (uint32_t jt = 0; jt < g.ext_t; ++jt) {
            for (uint32_t jh = 0; jh < g.ext_h; ++jh) {
                for (uint32_t jw = 0; jw < g.ext_w; ++jw) {
                    g.box_idx[static_cast<size_t>(b) * g.box_vol + c] = ((w0 + jw) * H + (h0 + jh)) * T + (t0 + jt);
                    ++c;
                }
            }
        }
        const std::array<int32_t, 3> key = {
            static_cast<int32_t>(t0) - (static_cast<int32_t>(bti * bt) - static_cast<int32_t>(ht)),
            static_cast<int32_t>(h0) - (static_cast<int32_t>(bhi * bh) - static_cast<int32_t>(hh)),
            static_cast<int32_t>(w0) - (static_cast<int32_t>(bwi * bw) - static_cast<int32_t>(hw))};
        auto it = combos.find(key);
        if (it == combos.end()) {
            it = combos.emplace(key, std::make_pair(static_cast<uint32_t>(combos.size()), b)).first;
        }
        g.mask_id[b] = it->second.first;
    }
    g.n_distinct = static_cast<uint32_t>(combos.size());
    g.mask_table.assign(static_cast<size_t>(g.n_distinct) * g.vol * g.box_vol, 0);
    for (const auto& [key, rowrep] : combos) {
        const uint32_t row = rowrep.first, brep = rowrep.second;
        const uint32_t bti = brep / (Hb * Wb), bhi = (brep / Wb) % Hb, bwi = brep % Wb;
        const uint32_t t0 = clamp_origin(bti, bt, ht, g.ext_t, T);
        const uint32_t h0 = clamp_origin(bhi, bh, hh, g.ext_h, H);
        const uint32_t w0 = clamp_origin(bwi, bw, hw, g.ext_w, W);
        for (uint32_t wid = 0; wid < g.vol; ++wid) {
            const uint32_t dt = wid / (bh * bw), dh = (wid / bw) % bh, dw = wid % bw;
            const uint32_t wt = nbr_start(bti * bt + dt, T, kt);
            const uint32_t wh = nbr_start(bhi * bh + dh, H, kh);
            const uint32_t ww = nbr_start(bwi * bw + dw, W, kw);
            uint32_t c = 0;
            for (uint32_t jt = 0; jt < g.ext_t; ++jt) {
                for (uint32_t jh = 0; jh < g.ext_h; ++jh) {
                    for (uint32_t jw = 0; jw < g.ext_w; ++jw) {
                        const bool live = (wt <= t0 + jt && t0 + jt < wt + ker_t) &&
                                          (wh <= h0 + jh && h0 + jh < wh + ker_h) &&
                                          (ww <= w0 + jw && w0 + jw < ww + ker_w);
                        g.mask_table[(static_cast<size_t>(row) * g.vol + wid) * g.box_vol + c] = live ? 1 : 0;
                        ++c;
                    }
                }
            }
        }
    }
    return g;
}

}  // namespace ttnn::operations::transformer::gna
