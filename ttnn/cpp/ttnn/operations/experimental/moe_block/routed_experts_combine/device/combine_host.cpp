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

RowPieces row_pieces(uint32_t W, std::optional<uint32_t> piece_max) {
    RowPieces p;
    p.row_bytes = W * 2;
    p.NT = (p.row_bytes + kTile - 1) / kTile;
    const uint32_t pm = piece_max.value_or(0) ? *piece_max : kPieceMax;
    p.NP = (p.row_bytes + pm - 1) / pm;
    p.PD = align64((p.row_bytes + p.NP - 1) / p.NP);
    p.PS = align64(p.PD + 32);
    p.slot = p.NP * p.PS;
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
    std::optional<uint32_t> piece_max,
    uint32_t local_keep) {
    const auto pc = row_pieces(W, piece_max);
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
    if (pc.NP != 1) {
        return fail(fmt::format(
            "the L1 hand-off takes one piece a row: piece {} < {} B", piece_max.value_or(kPieceMax), pc.row_bytes));
    }
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
    std::optional<uint32_t> piece_max,
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
                auto lay = combine_l1_layout(
                    N, S, W, static_cast<uint32_t>(A), arena_bytes, fused, qp, depth, piece_max, keep);
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

CreditPolicy credit_policy(
    uint32_t N, uint32_t S, uint32_t A, uint32_t PSL, const CombineTuning& tuning, uint32_t max_rows_per_token) {
    CreditPolicy p;
    // idle credit snapshots batched (at least 8 rows, or 6 us old) where there are many sources (Galaxy: -9 %)
    if (N >= 16) {
        p.cred_idle = 8;
        p.cred_age = 8100;
    }
    if (tuning.credidle) {
        p.cred_idle = tuning.credidle;
        p.cred_age = tuning.credage;
    }
    p.cred_thr = tuning.credthr;
    const uint32_t MAXT = (S + A - 1) / A;
    if (tuning.credauto && static_cast<uint64_t>(MAXT) * max_rows_per_token <= PSL) {
        p.cred_thr = 64;
        p.cred_idle = 8;
        p.cred_age = 4050;
        p.end_only = true;
    }
    return p;
}

uint32_t fused_arena_reserve(uint32_t flat_static_bytes) {
    return align64(flat_static_bytes) + kWriterWordsRows * 64 + 2048;
}

uint32_t exchange_arena_reserve(
    uint32_t flat_static_bytes, uint32_t dispatch_static_bytes, uint32_t manifest_bytes, bool dispatch_in_program) {
    if (dispatch_in_program) {
        return fused_arena_reserve(std::max(flat_static_bytes, dispatch_static_bytes)) + manifest_bytes + 1024;
    }
    return fused_arena_reserve(std::max(flat_static_bytes, manifest_bytes + dispatch_static_bytes)) + 1024;
}

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
           align64(3 * 2 * manifest_batch * 16) + align64(64);
}

PlanTables plan_reference(
    const std::vector<uint16_t>& ids,
    const std::vector<uint16_t>& wbits,
    uint32_t me,
    uint32_t R,
    uint32_t C,
    uint32_t S,
    uint32_t K,
    uint32_t EPC,
    uint32_t NG,
    uint32_t A,
    uint32_t L,
    uint32_t P,
    const std::array<uint8_t, 8>& lut,
    uint32_t rows_cap,
    bool wrap_x,
    bool wrap_y,
    bool token_major) {
    const uint32_t N = R * C;
    TT_FATAL(ids.size() == static_cast<size_t>(N) * S * K && wbits.size() == ids.size(), "plan_reference: [N, S, K]");
    const auto topo = MeshTopology{R, C, wrap_x, wrap_y};
    const uint32_t E0 = me * EPC;
    std::vector<uint32_t> lc(EPC, 0);
    for (const uint16_t g : ids) {
        if (g >= E0 && g < E0 + EPC) {
            ++lc[g - E0];
        }
    }
    PlanTables out;
    out.counts.assign(NG, 0);
    out.regions.assign(NG, 0);
    out.token_index.assign(rows_cap, 0);
    out.table.assign(2ull * rows_cap, 0);
    out.plan.assign(kPlanWords, 0);
    std::vector<uint32_t> fill(EPC, 0);
    uint32_t off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        out.counts[E0 + e] = lc[e];
        out.regions[E0 + e] = off;
        fill[e] = off;
        off += (lc[e] + 31) / 32 * 32;
    }
    out.plan[kPlanRows] = off;
    out.plan[kPlanOver] = off > rows_cap ? 1 : 0;
    auto at = [&](uint32_t o, uint32_t t, uint32_t k) { return (static_cast<size_t>(o) * S + t) * K + k; };
    auto visit = [&](uint32_t o, uint32_t t) {
        const uint32_t dh = o == me ? 0 : first_hop_idx(me, o, topo);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t g = ids[at(o, t, k)], wb = wbits[at(o, t, k)];
            const uint32_t src = g / EPC;
            const uint32_t l = link_of(t % A, src, o, L);
            if (src == me) {
                const uint32_t e = g - E0;
                const uint32_t r = fill[e]++;
                if (r < rows_cap) {
                    out.token_index[r] = o * S + t;
                    uint32_t dest = 0;
                    if (o == me) {
                        dest = P + t % A;
                    } else {
                        dest = lut[dh * 2 + l];
                        out.plan[kPlanNinj + dest] += 1;
                    }
                    out.table[2 * r] = dest | (wb << 16);
                    out.table[2 * r + 1] = o | (t << 8) | (e << 24);
                }
            }
            if (o == me) {
                if (src == me) {
                    out.plan[kPlanNloc + t % A] += 1;
                } else {
                    out.plan[kPlanWant + (t % A) * 32 + src] += 1;
                    out.plan[kPlanNret + lut[first_hop_idx(me, src, topo) * 2 + l]] += 1;
                }
            }
        }
    };
    if (token_major) {
        for (uint32_t t = 0; t < S; ++t) {
            for (uint32_t o = 0; o < N; ++o) {
                visit(o, t);
            }
        }
    } else {
        for (uint32_t o = 0; o < N; ++o) {
            for (uint32_t t = 0; t < S; ++t) {
                visit(o, t);
            }
        }
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
