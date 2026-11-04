// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_host.hpp"

#include <algorithm>
#include <map>
#include <tuple>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

RowSlot row_slot(uint32_t W) {
    RowSlot p;
    p.row_bytes = W * 2;
    p.NT = (p.row_bytes + kTile - 1) / kTile;
    p.slot = align64(align64(p.row_bytes) + 32);
    return p;
}

StateLayout state_layout(uint32_t N, uint32_t A) {
    const uint32_t AP = (A + 3) / 4 * 4;
    const uint32_t rw = align64(4ull * N * AP);
    StateLayout s;
    s.CRED = 0;
    s.USED = rw;
    s.RET = 2 * rw;
    s.LAYR = 3 * rw;
    s.LAYO = 4 * rw;
    s.U0 = 5 * rw;
    s.CNT = 6 * rw;
    s.MISC = s.CNT + 512;
    s.HDRS = s.MISC + kMiscBytes;
    s.RINGS = align64(s.HDRS + N * kNumHeaders * kHeaderStride + kNumHeaders * 4 * AP);
    return s;
}

uint32_t acc_cb_bytes(uint32_t S, uint32_t A, uint32_t NT) {
    const uint32_t MAXT = (S + A - 1) / A;
    return align64(kTabCbBytes) + align64(2 * NT * kTile) + align64(2 * 32) + align64(MAXT * NT * 4096) +
           align64(2 * NT * kTile) + align64(2 * kTile);
}

CombineLayout combine_l1_layout(
    uint32_t N,
    uint32_t S,
    uint32_t W,
    uint32_t A,
    uint32_t arena_bytes,
    bool acc_on_ports,
    uint32_t inj_slots,
    uint32_t depth,
    uint32_t local_keep) {
    const auto pc = row_slot(W);
    const int64_t ISLOT = pc.slot, SLOT = pc.slot;
    const int64_t RINGS = state_layout(N, A).RINGS;
    const uint32_t MAXT = (S + A - 1) / A;
    CombineLayout out;
    out.slot = pc.slot;
    out.acc_fp32 = MAXT * pc.NT * 4096;
    out.acc_cbs = acc_cb_bytes(S, A, pc.NT);
    out.rings = static_cast<uint32_t>(RINGS);
    const int64_t arena = arena_bytes, acc_cbs = out.acc_cbs;
    auto fail = [&](std::string why) {
        out.fits = false;
        out.why = std::move(why);
        return out;
    };
    int64_t QP = inj_slots;
    const int64_t keep = acc_on_ports ? (local_keep + std::min<int64_t>(32, QP)) * ISLOT : local_keep * ISLOT;
    // (Python floor division: a negative budget floors toward -inf)
    auto fdiv = [](int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    const int64_t fit = fdiv(arena - RINGS - acc_cbs - keep, SLOT);
    const int64_t PSL = std::min<int64_t>(static_cast<int64_t>(depth) * (N - 1), fit);
    out.PSL = static_cast<int32_t>(PSL);
    out.D = N > 1 ? static_cast<int32_t>(fdiv(PSL, N - 1)) : 0;
    out.has_PSL = true;
    if (PSL < static_cast<int64_t>(N) - 1) {
        return fail(fmt::format("landing pool {} < {} slots (one per source chip)", PSL, N - 1));
    }
    if (PSL >= 4096) {
        return fail("landing pool >= 4096 slots (12-bit ring fields)");
    }
    const int64_t acc_area = RINGS + PSL * SLOT;
    const int64_t a_end = align64(acc_area) + acc_cbs;
    int64_t QL = 0, acc_used = 0, port_used = 0;
    if (acc_on_ports) {
        QL = std::min<int64_t>(16, fdiv(arena - a_end - std::min<int64_t>(32, QP) * ISLOT, ISLOT));
        QP = std::min<int64_t>(QP, fdiv(arena - a_end - QL * ISLOT, ISLOT));
        if (QL < 2 || QP < 8) {
            return fail(fmt::format("acc_on_ports: local ring {} / injection ring {} slots", QL, QP));
        }
        acc_used = a_end + QL * ISLOT + QP * ISLOT;
        port_used = acc_used;
    } else {
        QL = std::min<int64_t>(16, fdiv(arena - a_end, ISLOT));
        if (QL < 2) {
            return fail(fmt::format("no room for the accumulators' local ring ({} slots)", QL));
        }
        acc_used = a_end + QL * ISLOT;
        port_used = align64(RINGS + QP * ISLOT) + kTabCbBytes;
        if (port_used > arena) {
            return fail(fmt::format("port: {} KB > the arena's {} KB", port_used >> 10, arena >> 10));
        }
    }
    out.QP = static_cast<uint32_t>(QP);
    out.QL = static_cast<uint32_t>(QL);
    out.acc_used = static_cast<uint32_t>(acc_used);
    out.port_used = static_cast<uint32_t>(port_used);
    out.has_rings = true;
    return out;
}

CombineAuto combine_auto(
    uint32_t N,
    uint32_t S,
    uint32_t W,
    uint32_t n_cells,
    uint32_t port_cells,
    uint32_t arena_bytes,
    uint32_t depth,
    uint32_t a_max,
    std::optional<CombineMode> mode) {
    CombineAuto res;
    std::vector<std::string> why;
    auto score = [](const CombineAutoChoice& c) {
        return std::make_tuple(c.layout.D, c.inj_slots, c.local_keep, c.layout.PSL);
    };
    for (const bool fused : {false, true}) {
        const int64_t cells = fused ? static_cast<int64_t>(n_cells) : static_cast<int64_t>(n_cells) - port_cells;
        const int64_t A = std::min<int64_t>(a_max, cells);
        const char* name = fused ? "acc_on_ports" : "ports apart";
        if (mode.has_value() && *mode != (fused ? CombineMode::AccOnPorts : CombineMode::PortsApart)) {
            continue;
        }
        if (A < 1 || (fused && static_cast<int64_t>(port_cells) > A)) {
            why.push_back(fmt::format("{}: {} accumulator cells", name, A));
            continue;
        }
        std::optional<CombineAutoChoice> mode_best;
        CombineLayout last;
        for (const uint32_t keep : {8u, 2u}) {
            for (const uint32_t qp : {64u, 32u, 16u, 8u}) {
                auto lay = combine_l1_layout(N, S, W, static_cast<uint32_t>(A), arena_bytes, fused, qp, depth, keep);
                if (!lay.fits) {
                    last = lay;
                    continue;
                }
                CombineAutoChoice cand{fused, static_cast<uint32_t>(A), qp, depth, keep, lay};
                if (!mode_best || score(cand) > score(*mode_best)) {
                    mode_best = cand;
                }
            }
        }
        if (!mode_best) {
            why.push_back(fmt::format("{} A {}: {} at the shallowest rings", name, A, last.why));
            continue;
        }
        (fused ? res.acc_on_ports : res.ports_apart) = mode_best;
        if (!res.best || mode_best->layout.D > res.best->layout.D) {
            res.best = mode_best;
        }
    }
    res.fits = res.best.has_value();
    if (!res.fits) {
        for (size_t i = 0; i < why.size(); ++i) {
            res.why += (i ? "; " : "") + why[i];
        }
    }
    return res;
}

CreditPolicy credit_policy(uint32_t N) {
    CreditPolicy p;
    // idle credit snapshots batched (at least 8 rows, or 6 us old) where there are many sources (Galaxy: -9 %)
    if (N >= 16) {
        p.cred_idle = 8;
        p.cred_age = 8100;
    }
    return p;
}

uint32_t fused_arena_reserve(uint32_t flat_static_bytes) {
    return align64(flat_static_bytes) + kWriterWordsRows * 64 + 2048;
}

uint32_t exchange_arena_reserve(uint32_t static_cb_bytes) { return fused_arena_reserve(static_cb_bytes) + 1024; }

uint32_t exchange_arena_bytes(uint32_t largest_free_bytes, uint32_t reserve) {
    const int64_t rows = (static_cast<int64_t>(largest_free_bytes) - reserve) / 64 / 32 * 32 - kWriterWordsRows;
    TT_FATAL(
        rows > 0,
        "routed_experts_combine: no L1 left for the arena ({} B free, {} B reserve)",
        largest_free_bytes,
        reserve);
    return static_cast<uint32_t>(rows * 64);
}

uint32_t dispatch_static_cb_bytes(
    uint32_t N, uint32_t S, uint32_t row_width, uint32_t group, uint32_t manifest_batch, uint32_t dtype_bytes) {
    constexpr uint32_t kMetaPage = 16;
    const uint32_t row_bytes = align64(row_width * dtype_bytes);
    const uint32_t plan_bytes = align64(S * route_words(N) * 4);
    return align64(2 * group * row_bytes) + align64(2 * group * kMetaPage) + align64(plan_bytes) +
           align64(3 * 2 * manifest_batch * 16);
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
