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

namespace ttnn::operations::bringup::moe_block::routed_experts_combine {

RowSlot row_slot(uint32_t W) {
    RowSlot p;
    p.row_bytes = W * 2;
    p.row_tiles = (p.row_bytes + kTile - 1) / kTile;
    p.slot_bytes = align64(align64(p.row_bytes) + 32);
    return p;
}

StateLayout state_layout(uint32_t N, uint32_t A, uint32_t grants_per_source) {
    const uint32_t accumulators_padded = (A + 3) / 4 * 4;
    const uint32_t region_bytes = align64(4ull * N * accumulators_padded);
    StateLayout s;
    s.grants_per_source = grants_per_source;
    s.credits_received_offset = 0;
    s.rows_sent_offset = region_bytes;
    s.credits_to_return_offset = 2 * region_bytes;
    s.ring_layouts_received_offset = 3 * region_bytes;
    s.ring_layouts_to_forward_offset = 4 * region_bytes;
    s.rows_sent_at_call_start_offset = 5 * region_bytes;
    s.counts_offset = 6 * region_bytes;
    if (grants_per_source) {  // (shared landing: the grant rings, before the counts)
        s.grant_ids_offset = s.counts_offset;
        s.grant_snapshots_offset = align64(s.grant_ids_offset + 4ull * N * accumulators_padded * grants_per_source);
        s.counts_offset = align64(s.grant_snapshots_offset + 4ull * N * accumulators_padded * (1 + grants_per_source));
    }
    s.misc_words_offset = s.counts_offset + 512;
    s.row_headers_offset = s.misc_words_offset + kMiscBytes;
    s.rings_offset = align64(
        s.row_headers_offset + N * kNumHeaders * kHeaderStride +
        kNumHeaders * 4 * accumulators_padded * (1 + grants_per_source));
    return s;
}

uint32_t acc_cb_bytes(uint32_t S, uint32_t A, uint32_t row_tiles, uint32_t rows, uint32_t tab_cb_bytes) {
    const uint32_t max_tokens_per_accumulator = (S + A - 1) / A;
    return align64(tab_cb_bytes) + align64(rows * row_tiles * kTile) + align64(rows * 32) +
           align64(max_tokens_per_accumulator * row_tiles * 4096) + align64(2 * row_tiles * kTile) +
           align64(rows * kTile);
}

CombineLayout combine_l1_layout(
    uint32_t N,
    uint32_t S,
    uint32_t W,
    uint32_t A,
    uint32_t arena_bytes,
    bool accumulators_on_ports,
    uint32_t injection_ring_slots_wanted,
    uint32_t depth,
    uint32_t local_ring_slots_reserved,
    bool shared_landing,
    uint32_t pool_slots) {
    const auto slot_sizes = row_slot(W);
    const int64_t slot_bytes = slot_sizes.slot_bytes;
    const int64_t state_bytes = state_layout(N, A, shared_landing ? kGrants : 0).rings_offset;
    const uint32_t max_tokens_per_accumulator = (S + A - 1) / A;
    CombineLayout out;
    out.slot_bytes = slot_sizes.slot_bytes;
    out.accumulator_fp32_bytes = max_tokens_per_accumulator * slot_sizes.row_tiles * 4096;
    out.accumulator_cb_bytes = acc_cb_bytes(S, A, slot_sizes.row_tiles, pool_slots > 0 ? kPoolRows : 2);
    out.state_bytes = static_cast<uint32_t>(state_bytes);
    const int64_t arena = arena_bytes;
    int64_t accumulator_cb_bytes = out.accumulator_cb_bytes;
    auto fail = [&](std::string why) {
        out.fits = false;
        out.why = std::move(why);
        return out;
    };
    int64_t injection_ring_slots = injection_ring_slots_wanted;
    const int64_t kept = accumulators_on_ports
                             ? (local_ring_slots_reserved + std::min<int64_t>(32, injection_ring_slots)) * slot_bytes
                             : local_ring_slots_reserved * slot_bytes;
    // (Python floor division: a negative budget floors toward -inf)
    auto fdiv = [](int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    // (the landing pool on the flat cores: each accumulator's share of their slots, nothing in its own L1)
    const bool landing_on_flat_expert_cores = pool_slots > 0;
    const int64_t fit = landing_on_flat_expert_cores
                            ? static_cast<int64_t>(pool_slots / A)
                            : fdiv(arena - state_bytes - accumulator_cb_bytes - kept, slot_bytes);
    const int64_t landing_slots_cap = !shared_landing
                                          ? std::numeric_limits<int64_t>::max()
                                          : int64_t(landing_on_flat_expert_cores ? kSharedFlatPslMax : kSharedPslMax);
    const int64_t landing_slots_per_accumulator =
        std::min<int64_t>({static_cast<int64_t>(depth) * (N - 1), fit, landing_slots_cap});
    if (shared_landing && landing_on_flat_expert_cores) {  // (CB_TAB holds the pool's owners / free list: sized by it)
        out.accumulator_cb_bytes = acc_cb_bytes(
            S,
            A,
            slot_sizes.row_tiles,
            kPoolRows,
            acc_tab_bytes(true, static_cast<uint32_t>(landing_slots_per_accumulator)));
        accumulator_cb_bytes = out.accumulator_cb_bytes;
    }
    out.landing_slots_per_accumulator = static_cast<int32_t>(landing_slots_per_accumulator);
    out.landing_slots_per_source = N > 1 ? static_cast<int32_t>(fdiv(landing_slots_per_accumulator, N - 1)) : 0;
    out.has_landing_slots = true;
    if (landing_slots_per_accumulator < static_cast<int64_t>(N) - 1) {
        return fail(
            fmt::format("landing pool {} < {} slots (one per source chip)", landing_slots_per_accumulator, N - 1));
    }
    if (landing_slots_per_accumulator >= 4096) {
        return fail("landing pool >= 4096 slots (12-bit ring fields)");
    }
    const int64_t accumulator_area =
        state_bytes + (landing_on_flat_expert_cores ? 0 : landing_slots_per_accumulator * slot_bytes);
    const int64_t accumulator_area_end = align64(accumulator_area) + accumulator_cb_bytes;
    int64_t local_ring_slots = 0, accumulator_core_bytes = 0, port_core_bytes = 0;
    if (accumulators_on_ports) {
        local_ring_slots = std::min<int64_t>(
            16,
            fdiv(arena - accumulator_area_end - std::min<int64_t>(32, injection_ring_slots) * slot_bytes, slot_bytes));
        injection_ring_slots = std::min<int64_t>(
            injection_ring_slots, fdiv(arena - accumulator_area_end - local_ring_slots * slot_bytes, slot_bytes));
        if (local_ring_slots < 2 || injection_ring_slots < 8) {
            return fail(fmt::format(
                "accumulators_on_ports: local ring {} / injection ring {} slots",
                local_ring_slots,
                injection_ring_slots));
        }
        accumulator_core_bytes =
            accumulator_area_end + local_ring_slots * slot_bytes + injection_ring_slots * slot_bytes;
        port_core_bytes = accumulator_core_bytes;
    } else {
        local_ring_slots = std::min<int64_t>(16, fdiv(arena - accumulator_area_end, slot_bytes));
        if (local_ring_slots < 2) {
            return fail(fmt::format("no room for the accumulators' local ring ({} slots)", local_ring_slots));
        }
        accumulator_core_bytes = accumulator_area_end + local_ring_slots * slot_bytes;
        port_core_bytes = align64(state_bytes + injection_ring_slots * slot_bytes) + kTabCbBytes;
        if (port_core_bytes > arena) {
            return fail(fmt::format("port: {} KB > the arena's {} KB", port_core_bytes >> 10, arena >> 10));
        }
    }
    out.injection_ring_slots = static_cast<uint32_t>(injection_ring_slots);
    out.local_ring_slots = static_cast<uint32_t>(local_ring_slots);
    out.accumulator_core_bytes = static_cast<uint32_t>(accumulator_core_bytes);
    out.port_core_bytes = static_cast<uint32_t>(port_core_bytes);
    out.has_ring_sizes = true;
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
    const bool shared_landing = landing_shared(landing);
    if (!landing_on_flat(landing)) {
        pool_slots = 0;
    }
    CombineAuto result;
    std::vector<std::string> why;
    auto score = [](const CombineAutoChoice& c) {
        return std::make_tuple(
            std::min<uint32_t>(c.layout.landing_slots_per_source, 3),
            std::min<uint32_t>(c.layout.injection_ring_slots, 64),
            c.layout.landing_slots_per_source,
            c.local_ring_slots_reserved,
            c.layout.landing_slots_per_accumulator);
    };
    for (const bool accumulators_on_ports : {false, true}) {
        const int64_t accumulator_core_budget = accumulators_on_ports
                                                    ? static_cast<int64_t>(free_core_count)
                                                    : static_cast<int64_t>(free_core_count) - port_core_count;
        // (the y writers address the chip's ports, then every accumulator: at most kHandoffDests destinations -
        // DeepSeek V4.1-Flash's narrower rows leave cores for 27 accumulators, 35 destinations with its 8 ports)
        const int64_t A = std::min<int64_t>(
            {static_cast<int64_t>(a_max), accumulator_core_budget, int64_t(kHandoffDests) - port_core_count});
        const char* name = accumulators_on_ports ? "accumulators_on_ports" : "ports apart";
        if (mode.has_value() && *mode != (accumulators_on_ports ? CombineMode::AccOnPorts : CombineMode::PortsApart)) {
            continue;
        }
        if (A < 1 || (accumulators_on_ports && static_cast<int64_t>(port_core_count) > A)) {
            why.push_back(fmt::format("{}: {} accumulator cores", name, A));
            continue;
        }
        std::optional<CombineAutoChoice> mode_best;
        CombineLayout last;
        for (const uint32_t local_reserve : {8u, 2u}) {
            for (const uint32_t injection_slots : {64u, 32u, 16u, 8u}) {
                auto layout = combine_l1_layout(
                    N,
                    S,
                    W,
                    static_cast<uint32_t>(A),
                    arena_bytes,
                    accumulators_on_ports,
                    injection_slots,
                    depth,
                    local_reserve,
                    shared_landing,
                    pool_slots);
                if (!layout.fits) {
                    last = layout;
                    continue;
                }
                CombineAutoChoice cand{
                    accumulators_on_ports,
                    static_cast<uint32_t>(A),
                    injection_slots,
                    depth,
                    local_reserve,
                    shared_landing,
                    layout};
                if (!mode_best || score(cand) > score(*mode_best)) {
                    mode_best = cand;
                }
            }
        }
        if (!mode_best) {
            why.push_back(fmt::format("{} A {}: {} at the shallowest rings", name, A, last.why));
            continue;
        }
        (accumulators_on_ports ? result.accumulators_on_ports : result.ports_apart) = mode_best;
    }
    for (const auto* m :
         {&result.ports_apart, &result.accumulators_on_ports}) {  // (the modes ranked as combine_choice_key)
        if (!m->has_value()) {
            continue;
        }
        uint32_t a_ref = (*m)->accumulator_count;
        for (const auto* o : {&result.ports_apart, &result.accumulators_on_ports}) {
            a_ref = o->has_value() ? std::max(a_ref, (*o)->accumulator_count) : a_ref;
        }
        if (!result.best || combine_choice_key(**m, a_ref) > combine_choice_key(*result.best, a_ref)) {
            result.best = *m;
        }
    }
    result.fits = result.best.has_value();
    if (!result.fits) {
        for (size_t i = 0; i < why.size(); ++i) {
            result.why += (i ? "; " : "") + why[i];
        }
    }
    return result;
}

std::tuple<uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t> combine_choice_key(
    const CombineAutoChoice& c, uint32_t a_ref) {
    return std::make_tuple(
        std::min<uint32_t>(c.layout.landing_slots_per_source, 3),
        2 * c.accumulator_count >= a_ref,
        std::min<uint32_t>(c.layout.injection_ring_slots, 64),
        static_cast<uint32_t>(c.layout.landing_slots_per_source),
        c.accumulator_count,
        c.layout.local_ring_slots);
}

CreditPolicy credit_policy(
    uint32_t N, uint32_t landing_slots_per_source, bool shared_landing, std::optional<uint32_t> batch) {
    // With > 16 chips (a Galaxy) idle snapshots batched, 8 rows or 6 us old: the snapshots' own traffic counts there
    // (GLM-5.3 at its Galaxy D 2, 31 sources emulated: accumulators on the ports 836 -> 762 us). The rings on 16 chips
    // (a Galaxy's 4x4 sub-torus) too: 4x4 GLM-5.3 chunk 10240 D 2 combine 2.915 ms at a batch of 1 (D / 2), 2.782 at
    // 2, 2.774 at 4, 2.735 at 8; chunk 7680 D 3: 2.198 / 2.117 / 2.114 at 1 / 4 / 8. (The shared landing there stays at
    // 1: 4x4 chunk 7680 D 1 ~4 % slower at 8, chunk 5120 D 3 ~2-3 % faster at 2-8.) With fewer, a batch is never more
    // than half a source's slots - a source with D slots waited for 8 consumed rows, or the age (2x4 GLM-5.3-Flash 1280
    // rings experts + combine 723 -> 532 us at D 2, 1307 -> 888 at D 1, 515 at D 8 either way); shared: a grant is on
    // its source's critical path - every snapshot at once (688 -> 546 us at D 2, 1172 -> 800 at D 1).
    CreditPolicy p;
    const uint32_t b = batch                                      ? std::max<uint32_t>(1, *batch)
                       : (N > 16 || (N == 16 && !shared_landing)) ? 8u
                       : shared_landing                           ? 1u
                                        : std::clamp<uint32_t>(landing_slots_per_source / 2, 1, 8);
    p.return_credits_after_rows = b;
    p.idle_snapshot_rows = b;
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

}  // namespace ttnn::operations::bringup::moe_block::routed_experts_combine
