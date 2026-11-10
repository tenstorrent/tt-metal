// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_host.hpp"

#include <algorithm>
#include <limits>
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

StateLayout state_layout(uint32_t N, uint32_t A, uint32_t G) {
    const uint32_t AP = (A + 3) / 4 * 4;
    const uint32_t rw = align64(4ull * N * AP);
    StateLayout s;
    s.G = G;
    s.CRED = 0;
    s.USED = rw;
    s.RET = 2 * rw;
    s.LAYR = 3 * rw;
    s.LAYO = 4 * rw;
    s.U0 = 5 * rw;
    s.CNT = 6 * rw;
    if (G) {  // (shared landing: the grant rings, before the counts)
        s.GIDH = s.CNT;
        s.GREC = align64(s.GIDH + 4ull * N * AP * G);
        s.CNT = align64(s.GREC + 4ull * N * AP * (1 + G));
    }
    s.MISC = s.CNT + 512;
    s.HDRS = s.MISC + kMiscBytes;
    s.RINGS = align64(s.HDRS + N * kNumHeaders * kHeaderStride + kNumHeaders * 4 * AP * (1 + G));
    return s;
}

uint32_t acc_cb_bytes(uint32_t S, uint32_t A, uint32_t NT, uint32_t rows, uint32_t tab) {
    const uint32_t MAXT = (S + A - 1) / A;
    return align64(tab) + align64(rows * NT * kTile) + align64(rows * 32) + align64(MAXT * NT * 4096) +
           align64(2 * NT * kTile) + align64(rows * kTile);
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
    uint32_t local_keep,
    bool shared,
    uint32_t pool_slots) {
    const auto pc = row_slot(W);
    const int64_t ISLOT = pc.slot, SLOT = pc.slot;
    const int64_t RINGS = state_layout(N, A, shared ? kGrants : 0).RINGS;
    const uint32_t MAXT = (S + A - 1) / A;
    CombineLayout out;
    out.slot = pc.slot;
    out.acc_fp32 = MAXT * pc.NT * 4096;
    out.acc_cbs = acc_cb_bytes(S, A, pc.NT, pool_slots > 0 ? kPoolRows : 2);
    out.rings = static_cast<uint32_t>(RINGS);
    const int64_t arena = arena_bytes;
    int64_t acc_cbs = out.acc_cbs;
    auto fail = [&](std::string why) {
        out.fits = false;
        out.why = std::move(why);
        return out;
    };
    int64_t QP = inj_slots;
    const int64_t keep = acc_on_ports ? (local_keep + std::min<int64_t>(32, QP)) * ISLOT : local_keep * ISLOT;
    // (Python floor division: a negative budget floors toward -inf)
    auto fdiv = [](int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    // (the landing pool on the flat cores: each accumulator's share of their slots, nothing in its own L1)
    const bool flat_pool = pool_slots > 0;
    const int64_t fit = flat_pool ? static_cast<int64_t>(pool_slots / A) : fdiv(arena - RINGS - acc_cbs - keep, SLOT);
    const int64_t psl_cap =
        !shared ? std::numeric_limits<int64_t>::max() : int64_t(flat_pool ? kSharedFlatPslMax : kSharedPslMax);
    const int64_t PSL = std::min<int64_t>({static_cast<int64_t>(depth) * (N - 1), fit, psl_cap});
    if (shared && flat_pool) {  // (CB_TAB holds the pool's owners / free list: sized by it)
        out.acc_cbs = acc_cb_bytes(S, A, pc.NT, kPoolRows, acc_tab_bytes(true, static_cast<uint32_t>(PSL)));
        acc_cbs = out.acc_cbs;
    }
    out.PSL = static_cast<int32_t>(PSL);
    out.D = N > 1 ? static_cast<int32_t>(fdiv(PSL, N - 1)) : 0;
    out.has_PSL = true;
    if (PSL < static_cast<int64_t>(N) - 1) {
        return fail(fmt::format("landing pool {} < {} slots (one per source chip)", PSL, N - 1));
    }
    if (PSL >= 4096) {
        return fail("landing pool >= 4096 slots (12-bit ring fields)");
    }
    const int64_t acc_area = RINGS + (flat_pool ? 0 : PSL * SLOT);
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
    uint32_t free_core_count,
    uint32_t port_core_count,
    uint32_t arena_bytes,
    uint32_t depth,
    uint32_t a_max,
    std::optional<CombineMode> mode,
    CombineLanding landing,
    uint32_t pool_slots) {
    const bool shared = landing_shared(landing);
    if (!landing_on_flat(landing)) {
        pool_slots = 0;
    }
    CombineAuto res;
    std::vector<std::string> why;
    auto score = [](const CombineAutoChoice& c) {
        return std::make_tuple(
            std::min<uint32_t>(c.layout.D, 3),
            std::min<uint32_t>(c.layout.QP, 64),
            c.layout.D,
            c.local_keep,
            c.layout.PSL);
    };
    for (const bool acc_on_ports : {false, true}) {
        const int64_t accumulator_core_budget = acc_on_ports ? static_cast<int64_t>(free_core_count)
                                                             : static_cast<int64_t>(free_core_count) - port_core_count;
        // (the y writers address the chip's ports, then every accumulator: at most kHandoffDests destinations -
        // DeepSeek V4.1-Flash's narrower rows leave cores for 27 accumulators, 35 destinations with its 8 ports)
        const int64_t A = std::min<int64_t>(
            {static_cast<int64_t>(a_max), accumulator_core_budget, int64_t(kHandoffDests) - port_core_count});
        const char* name = acc_on_ports ? "acc_on_ports" : "ports apart";
        if (mode.has_value() && *mode != (acc_on_ports ? CombineMode::AccOnPorts : CombineMode::PortsApart)) {
            continue;
        }
        if (A < 1 || (acc_on_ports && static_cast<int64_t>(port_core_count) > A)) {
            why.push_back(fmt::format("{}: {} accumulator cores", name, A));
            continue;
        }
        std::optional<CombineAutoChoice> mode_best;
        CombineLayout last;
        for (const uint32_t keep : {8u, 2u}) {
            for (const uint32_t qp : {64u, 32u, 16u, 8u}) {
                auto lay = combine_l1_layout(
                    N, S, W, static_cast<uint32_t>(A), arena_bytes, acc_on_ports, qp, depth, keep, shared, pool_slots);
                if (!lay.fits) {
                    last = lay;
                    continue;
                }
                CombineAutoChoice cand{acc_on_ports, static_cast<uint32_t>(A), qp, depth, keep, shared, lay};
                if (!mode_best || score(cand) > score(*mode_best)) {
                    mode_best = cand;
                }
            }
        }
        if (!mode_best) {
            why.push_back(fmt::format("{} A {}: {} at the shallowest rings", name, A, last.why));
            continue;
        }
        (acc_on_ports ? res.acc_on_ports : res.ports_apart) = mode_best;
    }
    for (const auto* m : {&res.ports_apart, &res.acc_on_ports}) {  // (the modes ranked as combine_choice_key)
        if (!m->has_value()) {
            continue;
        }
        uint32_t a_ref = (*m)->n_acc;
        for (const auto* o : {&res.ports_apart, &res.acc_on_ports}) {
            a_ref = o->has_value() ? std::max(a_ref, (*o)->n_acc) : a_ref;
        }
        if (!res.best || combine_choice_key(**m, a_ref) > combine_choice_key(*res.best, a_ref)) {
            res.best = *m;
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

std::tuple<uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t> combine_choice_key(
    const CombineAutoChoice& c, uint32_t a_ref) {
    return std::make_tuple(
        std::min<uint32_t>(c.layout.D, 3),
        2 * c.n_acc >= a_ref,
        std::min<uint32_t>(c.layout.QP, 64),
        static_cast<uint32_t>(c.layout.D),
        c.n_acc,
        c.layout.QL);
}

CreditPolicy credit_policy(uint32_t N, uint32_t D, bool shared, std::optional<uint32_t> batch) {
    // With > 16 chips (a Galaxy) idle snapshots batched, 8 rows or 6 us old: the snapshots' own traffic counts there
    // (GLM-5.3 at its Galaxy D 2, 31 sources emulated: acc_on_ports 836 -> 762 us). The rings on 16 chips (a Galaxy's
    // 4x4 sub-torus) too: 4x4 GLM-5.3 chunk 10240 D 2 combine 2.915 ms at a batch of 1 (D / 2), 2.782 at 2, 2.774 at 4,
    // 2.735 at 8; chunk 7680 D 3: 2.198 / 2.117 / 2.114 at 1 / 4 / 8. (The shared landing there stays at 1: 4x4 chunk
    // 7680 D 1 ~4 % slower at 8, chunk 5120 D 3 ~2-3 % faster at 2-8.) With fewer, a batch is never more
    // than half a source's slots - a source with D slots waited for 8 consumed rows, or the age (2x4 GLM-5.3-Flash 1280
    // rings experts + combine 723 -> 532 us at D 2, 1307 -> 888 at D 1, 515 at D 8 either way); shared: a grant is on
    // its source's critical path - every snapshot at once (688 -> 546 us at D 2, 1172 -> 800 at D 1).
    CreditPolicy p;
    const uint32_t b = batch ? std::max<uint32_t>(1, *batch)
                       : (N > 16 || (N == 16 && !shared)) ? 8u
                       : shared ? 1u
                                : std::clamp<uint32_t>(D / 2, 1, 8);
    p.cred_thr = b;
    p.cred_idle = b;
    return p;
}

uint32_t acc_on_ports_arena_reserve(uint32_t flat_static_bytes) {
    return align64(flat_static_bytes) + kWriterWordsRows * 64 + 2048;
}

uint32_t exchange_arena_reserve(uint32_t static_cb_bytes) { return acc_on_ports_arena_reserve(static_cb_bytes) + 1024; }

uint32_t exchange_arena_bytes(uint32_t largest_free_bytes, uint32_t reserve) {
    const int64_t rows =
        (static_cast<int64_t>(largest_free_bytes) - reserve) / 64 / 32 * 32 - kWriterWordsRows - kArgsRows;
    TT_FATAL(
        rows > 0,
        "routed_experts_combine: no L1 left for the arena ({} B free, {} B reserve)",
        largest_free_bytes,
        reserve);
    return static_cast<uint32_t>(rows * 64);
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
