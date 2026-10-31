// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side K/V chain construction shared by the ring joint SDPA and sparse ring SDPA program factories: each core's
// flat Q range, linear store-and-forward chains, and row-wide multicast groups for the shared-K and GQA K/V chains.
// The chain arguments' layout is in kernels/ring_joint_chain_layout.hpp.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct CoreHeadWork {
    uint32_t head = 0;
    uint32_t q_chunk_count = 0;
};

struct CoreWork {
    CoreCoord physical_core;
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;
    std::vector<CoreHeadWork> head_work;
};

struct HeadSegmentRef {
    uint32_t core_idx = 0;
    uint32_t head_work_index = 0;
};

// Unified chain configuration for head-level and shared-K chains.
struct ChainConfig {
    // Core participation flags
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;

    // Chain scope: head distinguishes head-level from shared-K chains.
    uint32_t head = 0;  // 0 for shared-K chains

    // Linear chain topology
    CoreCoord prev_physical = CoreCoord{0, 0};
    CoreCoord next_physical = CoreCoord{0, 0};
    uint32_t next_core_q_chunks = 0;

    // Multicast configuration (1D for V, 2D for K)
    CoreCoord mcast_start = CoreCoord{0, 0};        // Rectangle start (physical)
    CoreCoord mcast_end = CoreCoord{0, 0};          // Rectangle end (physical)
    CoreCoord injector_physical = CoreCoord{0, 0};  // Injector's coords (for receiver sem addr in mcast)
    uint32_t mcast_num_dests = 0;                   // Receivers count (excludes self)

    // Append runtime args in canonical order
    void append_to_args(std::vector<uint32_t>& args) const {
        const size_t start_size = args.size();
        args.push_back(static_cast<uint32_t>(participates));
        args.push_back(static_cast<uint32_t>(is_injector));
        args.push_back(static_cast<uint32_t>(is_sink));
        args.push_back(head);
        args.push_back(static_cast<uint32_t>(prev_physical.x));
        args.push_back(static_cast<uint32_t>(prev_physical.y));
        args.push_back(static_cast<uint32_t>(next_physical.x));
        args.push_back(static_cast<uint32_t>(next_physical.y));
        args.push_back(next_core_q_chunks);
        args.push_back(static_cast<uint32_t>(mcast_start.x));
        args.push_back(static_cast<uint32_t>(mcast_start.y));
        args.push_back(static_cast<uint32_t>(mcast_end.x));
        args.push_back(static_cast<uint32_t>(mcast_end.y));
        args.push_back(static_cast<uint32_t>(injector_physical.x));
        args.push_back(static_cast<uint32_t>(injector_physical.y));
        args.push_back(mcast_num_dests);
        TT_FATAL(
            args.size() == start_size + kChainConfigRuntimeArgCount,
            "ChainConfig expected to append {} runtime args, appended {}",
            kChainConfigRuntimeArgCount,
            args.size() - start_size);
    }
};

// Head segments of the flat Q chunk range [flat_chunk, flat_chunk + remaining) of a core (num_q_chunks per head,
// NH heads per batch), appended to its head_work and, when head_segments is non-empty, to its head's segment list.
inline void append_head_work(
    std::vector<CoreWork>& core_work,
    std::vector<std::vector<HeadSegmentRef>>& head_segments,
    uint32_t core_idx,
    uint32_t flat_chunk,
    uint32_t remaining,
    uint32_t num_q_chunks,
    uint32_t NH) {
    auto& work = core_work.at(core_idx);
    while (remaining > 0) {
        const uint32_t head_index = num_q_chunks == 0 ? 0 : (flat_chunk / num_q_chunks);
        const uint32_t q_chunk_idx = num_q_chunks == 0 ? 0 : (flat_chunk % num_q_chunks);
        const uint32_t head_idx = (NH == 0) ? 0 : (head_index % NH);
        const uint32_t take = std::min(remaining, num_q_chunks - q_chunk_idx);
        work.head_work.push_back(CoreHeadWork{.head = head_idx, .q_chunk_count = take});
        if (!head_segments.empty()) {
            TT_FATAL(
                head_idx < head_segments.size(),
                "Head-chain segment index {} is outside {} configured query heads",
                head_idx,
                head_segments.size());
            head_segments[head_idx].push_back(HeadSegmentRef{
                .core_idx = core_idx, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
        }
        remaining -= take;
        flat_chunk += take;
    }
}

// Build a linear chain from sorted (core_idx, q_chunk_count) pairs.
// - chain_segs[i].second = q iterations the i-th core will process in this chain scope
// - injector = first core with head_work.size() == 1 (single head segment = no straddling)
// - no wrap-around: wrapping would inflate q_iter_local and cause deadlock
// - injector reselection for mcast is done separately in the mcast eligibility pass
using ChainSegment = std::pair<uint32_t, uint32_t>;  // (core_idx, q_chunk_count)
inline bool build_linear_chain(
    const std::vector<ChainSegment>& chain_segs,
    uint32_t head,
    std::vector<ChainConfig>& chain_configs,
    const std::vector<CoreWork>& core_work,
    bool require_single_head_injector = true) {
    if (chain_segs.size() < 2) {
        return false;
    }
    std::optional<size_t> injector_pos;
    for (size_t idx = 0; idx + 1 < chain_segs.size(); ++idx) {
        if (core_work[chain_segs[idx].first].global_q_count == 0) {
            continue;
        }
        if (!require_single_head_injector || core_work[chain_segs[idx].first].head_work.size() == 1) {
            injector_pos = idx;
            break;
        }
    }
    if (!injector_pos.has_value()) {
        return false;
    }
    const size_t start = *injector_pos;
    for (size_t idx = start; idx < chain_segs.size(); ++idx) {
        uint32_t ci = chain_segs[idx].first;
        auto& cfg = chain_configs[ci];
        cfg.participates = true;
        cfg.head = head;
        cfg.is_injector = (idx == start);
        cfg.is_sink = (idx == chain_segs.size() - 1);
        if (idx > start) {
            cfg.prev_physical = core_work[chain_segs[idx - 1].first].physical_core;
        }
        if (idx + 1 < chain_segs.size()) {
            cfg.next_physical = core_work[chain_segs[idx + 1].first].physical_core;
            cfg.next_core_q_chunks = chain_segs[idx + 1].second;
        }
    }
    return true;
}

struct RowWideChainMcastSelection {
    uint32_t row = 0;
    uint32_t injector_idx = 0;
    uint32_t max_q = 0;
    uint32_t num_receivers = 0;
    CoreCoord injector_physical = CoreCoord{0, 0};
    CoreCoord phys_start = CoreCoord{0, 0};
    CoreCoord phys_end = CoreCoord{0, 0};
};

// Pick a row-wide multicast injector among max-work cores. The recent-column FIFO spreads injectors across
// physical X columns for NoC diversity; correctness still requires a max-work injector so padded loops never
// read K/V beyond the real q-iteration span.
template <typename DeviceT>
std::optional<RowWideChainMcastSelection> select_row_wide_chain_mcast(
    const std::vector<CoreWork>& core_work,
    CoreCoord grid_size,
    DeviceT* device,
    uint32_t row,
    std::deque<uint32_t>& recent_cols) {
    if (recent_cols.size() >= grid_size.x) {
        recent_cols.pop_front();
    }

    uint32_t row_max_q = 0;
    for (uint32_t col = 0; col < grid_size.x; ++col) {
        const uint32_t ci = row * grid_size.x + col;
        row_max_q = std::max(row_max_q, core_work[ci].global_q_count);
    }
    if (row_max_q == 0) {
        return std::nullopt;
    }

    uint32_t injector_idx = std::numeric_limits<uint32_t>::max();
    for (uint32_t col = 0; col < grid_size.x; ++col) {
        const uint32_t ci = row * grid_size.x + col;
        if (core_work[ci].global_q_count != row_max_q) {
            continue;
        }
        const uint32_t phys_x = core_work[ci].physical_core.x;
        const bool excluded = std::find(recent_cols.begin(), recent_cols.end(), phys_x) != recent_cols.end();
        if (!excluded) {
            injector_idx = ci;
            break;
        }
        if (injector_idx == std::numeric_limits<uint32_t>::max()) {
            injector_idx = ci;
        }
    }

    TT_FATAL(
        injector_idx != std::numeric_limits<uint32_t>::max(),
        "Row mcast failed to find a max-work injector for row {}",
        row);
    const CoreCoord injector_physical = core_work[injector_idx].physical_core;
    recent_cols.push_back(injector_physical.x);

    return RowWideChainMcastSelection{
        .row = row,
        .injector_idx = injector_idx,
        .max_q = row_max_q,
        .num_receivers = static_cast<uint32_t>(grid_size.x - 1),
        .injector_physical = injector_physical,
        .phys_start = device->worker_core_from_logical_core(CoreCoord{0, row}),
        .phys_end = device->worker_core_from_logical_core(CoreCoord{grid_size.x - 1, row}),
    };
}

inline void configure_row_wide_chain_mcast(
    const RowWideChainMcastSelection& selection,
    uint32_t head,
    std::vector<ChainConfig>& chain_configs,
    std::vector<uint32_t>& chain_max_q,
    CoreCoord grid_size) {
    for (uint32_t col = 0; col < grid_size.x; ++col) {
        const uint32_t ci = selection.row * grid_size.x + col;
        auto& cfg = chain_configs[ci];
        cfg.participates = true;
        cfg.head = head;
        cfg.prev_physical = CoreCoord{0, 0};
        cfg.next_physical = CoreCoord{0, 0};
        cfg.mcast_start = selection.phys_start;
        cfg.mcast_end = selection.phys_end;
        cfg.injector_physical = selection.injector_physical;
        cfg.is_injector = (ci == selection.injector_idx);
        cfg.is_sink = !cfg.is_injector;
        if (cfg.is_injector) {
            cfg.mcast_num_dests = selection.num_receivers;
            cfg.next_core_q_chunks = selection.max_q;
        } else {
            cfg.mcast_num_dests = 0;
            cfg.next_core_q_chunks = 0;
        }
        chain_max_q[ci] = selection.max_q;
    }
}

struct GqaKvChains {
    uint32_t grouped_chains = 0;
    uint32_t grouped_participant_cores = 0;
    uint32_t local_fallback_cores = 0;
    bool mcast_enabled = false;
    std::string mcast_fallback_reason;
};

// GQA K/V chains: one linear chain per K/V head over the cores whose Q range stays in that head's group (other
// cores read their own K/V). With one local K/V head per chip (B = 1, NHK = NHV = 1) every active Q-head core
// consumes the same K and V chunks, so each row with work uses a row-wide multicast instead of the store-and-forward
// chain; idle cores in an active row participate in padded iterations. chain_max_q gets each multicast member's
// loop-padding count.
template <typename DeviceT>
GqaKvChains build_gqa_kv_chains(
    const std::vector<CoreWork>& core_work,
    std::vector<ChainConfig>& chain_configs,
    std::vector<uint32_t>& chain_max_q,
    uint32_t NHK,
    uint32_t q_heads_per_kv,
    CoreCoord grid_size,
    DeviceT* device) {
    GqaKvChains res;
    const uint32_t num_cores = static_cast<uint32_t>(core_work.size());
    std::vector<std::vector<ChainSegment>> kv_group_segments(NHK);

    for (uint32_t ci = 0; ci < num_cores; ++ci) {
        if (core_work[ci].global_q_count == 0) {
            continue;
        }

        bool has_single_group = false;
        bool spans_multiple_groups = false;
        uint32_t single_group_id = 0;
        uint32_t q_chunk_count = 0;
        for (const auto& hw : core_work[ci].head_work) {
            const uint32_t kv_head = hw.head / q_heads_per_kv;
            if (!has_single_group) {
                has_single_group = true;
                single_group_id = kv_head;
            } else if (single_group_id != kv_head) {
                spans_multiple_groups = true;
                break;
            }
            q_chunk_count += hw.q_chunk_count;
        }

        if (has_single_group && !spans_multiple_groups) {
            kv_group_segments[single_group_id].emplace_back(ci, q_chunk_count);
            res.grouped_participant_cores++;
        } else {
            res.local_fallback_cores++;
        }
    }

    for (uint32_t kv_head = 0; kv_head < static_cast<uint32_t>(kv_group_segments.size()); ++kv_head) {
        const auto& chain_segs = kv_group_segments[kv_head];
        if (chain_segs.size() < 2) {
            continue;
        }
        if (build_linear_chain(chain_segs, kv_head, chain_configs, core_work, false)) {
            res.grouped_chains++;
        }
    }

    if (NHK != 1) {
        res.mcast_fallback_reason = "NHK != 1 (multi-KV-head GQA mcast not supported)";
    } else if (num_cores < 2) {
        res.mcast_fallback_reason = "num_cores < 2";
    } else if (grid_size.x < 2) {
        res.mcast_fallback_reason = "grid_size.x < 2 (singleton rows)";
    } else {
        std::deque<uint32_t> recent_cols;  // FIFO of <= grid.x-1 most-recent claimed phys_x
        uint32_t gqa_mcast_rows = 0;

        for (uint32_t row = 0; row < grid_size.y; ++row) {
            const auto selection = select_row_wide_chain_mcast(core_work, grid_size, device, row, recent_cols);
            if (!selection.has_value()) {
                continue;
            }
            configure_row_wide_chain_mcast(*selection, 0, chain_configs, chain_max_q, grid_size);
            gqa_mcast_rows++;
            log_debug(
                tt::LogOp,
                "GQA K/V mcast row {}: injector core {} phys=({},{}) max_q={}, rect ({},{})-({},{})",
                selection->row,
                selection->injector_idx,
                selection->injector_physical.x,
                selection->injector_physical.y,
                selection->max_q,
                selection->phys_start.x,
                selection->phys_start.y,
                selection->phys_end.x,
                selection->phys_end.y);
        }

        res.mcast_enabled = gqa_mcast_rows > 0;
        if (!res.mcast_enabled) {
            res.mcast_fallback_reason = "no active groups";
        }
    }
    return res;
}

// Shared-K chain for separate-V / latent cases: K is shared across all heads, so all active cores form one chain,
// sorted by physical position for a stable unicast ordering (overwritten by the multicast pass if eligible).
inline void build_shared_k_chain(const std::vector<CoreWork>& core_work, std::vector<ChainConfig>& chain_configs) {
    std::vector<uint32_t> core_indices;
    for (uint32_t i = 0; i < static_cast<uint32_t>(core_work.size()); ++i) {
        if (core_work[i].global_q_count == 0) {
            continue;
        }
        core_indices.push_back(i);
    }

    std::sort(core_indices.begin(), core_indices.end(), [&](uint32_t a, uint32_t b) {
        const auto& pa = core_work[a].physical_core;
        const auto& pb = core_work[b].physical_core;
        return (pa.y < pb.y) || (pa.y == pb.y && pa.x < pb.x);
    });

    std::vector<ChainSegment> chain_segs;
    chain_segs.reserve(core_indices.size());
    for (uint32_t ci : core_indices) {
        chain_segs.emplace_back(ci, core_work[ci].global_q_count);
    }
    if (build_linear_chain(chain_segs, 0, chain_configs, core_work)) {
        log_debug(tt::LogOp, "K unicast chain: {} cores", chain_segs.size());
    }
}

// Shared-K multicast pass: one multicast chain per logical row, all or nothing: every row must contain work,
// otherwise the linear chain stays active. Returns whether it was enabled; fallback_reason says why not.
template <typename DeviceT>
bool build_shared_k_mcast(
    const std::vector<CoreWork>& core_work,
    std::vector<ChainConfig>& chain_configs,
    std::vector<uint32_t>& chain_max_q,
    CoreCoord grid_size,
    DeviceT* device,
    std::string& fallback_reason) {
    const uint32_t num_cores = static_cast<uint32_t>(core_work.size());
    if (num_cores < 2) {
        fallback_reason = "num_cores < 2";
        return false;
    }
    if (grid_size.x < 2) {
        // Each chain would be a singleton (1 core, no sinks) — mcast is degenerate.
        fallback_reason = "grid_size.x < 2 (singleton chains)";
        return false;
    }
    std::vector<RowWideChainMcastSelection> row_mcast_selections;
    row_mcast_selections.reserve(grid_size.y);
    std::deque<uint32_t> recent_cols;  // FIFO of <= grid.x-1 most-recent claimed phys_x
    for (uint32_t row = 0; row < grid_size.y; ++row) {
        const auto selection = select_row_wide_chain_mcast(core_work, grid_size, device, row, recent_cols);
        if (!selection.has_value()) {
            fallback_reason = fmt::format("row {} has no work", row);
            return false;
        }
        row_mcast_selections.push_back(*selection);
    }
    for (const auto& selection : row_mcast_selections) {
        configure_row_wide_chain_mcast(selection, 0, chain_configs, chain_max_q, grid_size);
        log_debug(
            tt::LogOp,
            "K mcast row {}: injector core {} phys=({},{}) max_q={}, rect ({},{})-({},{})",
            selection.row,
            selection.injector_idx,
            selection.injector_physical.x,
            selection.injector_physical.y,
            selection.max_q,
            selection.phys_start.x,
            selection.phys_start.y,
            selection.phys_end.x,
            selection.phys_end.y);
    }
    return true;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
