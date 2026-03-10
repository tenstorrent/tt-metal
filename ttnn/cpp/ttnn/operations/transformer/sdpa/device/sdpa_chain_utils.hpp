// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::prim::detail {

// ---- Chain management structures for KV store-and-forward optimization ----

struct CoreHeadWork {
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t q_chunk_start = 0;
    uint32_t q_chunk_count = 0;
};

struct CoreWork {
    CoreCoord logical_core;
    CoreCoord physical_core;
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;
    std::vector<CoreHeadWork> head_work;
};

struct HeadSegmentRef {
    uint32_t core_idx = 0;
    uint32_t head_work_index = 0;
};

struct CoreChainInfo {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t q_chunk_start = 0;
    uint32_t q_chunk_count = 0;
    CoreCoord prev_physical = CoreCoord{0, 0};
    CoreCoord next_physical = CoreCoord{0, 0};
    uint32_t next_core_q_chunks = 0;
    bool use_mcast = false;
    uint32_t mcast_num_dests = 0;    // num_dests for mcast API (includes self if injector inside rect)
    uint32_t mcast_sender_wait = 0;  // number of actual receivers that signal back (always chain_size - 1)
};

// ---- Chain builder for KV store-and-forward optimization ----
//
// For each head that spans >= 2 cores, builds a forwarding chain:
// find the best chain start, build wrap-around order, handle uniform/mixed
// q_chunks, and select the injector core by DRAM channel distance.
//
// `num_heads_per_batch` is NQH for classic SDPA or NH for ring-joint SDPA.
// It is used to decompose head_id -> head via `head_id % num_heads_per_batch`.
static inline void build_chains_for_heads(
    const std::vector<std::vector<HeadSegmentRef>>& head_segments,
    std::vector<CoreChainInfo>& core_chain_info,
    const std::vector<CoreWork>& core_work,
    uint32_t num_heads_per_batch,
    bool allow_wrap_back = true) {
    uint32_t chains_built = 0;
    uint32_t chains_skipped = 0;
    // Track injector physical X columns for DRAM channel spreading
    std::vector<uint32_t> injector_phys_x;

    for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
        auto& segments = head_segments[head_id];
        if (segments.size() < 2) {
            continue;  // No chain needed for single core
        }

        // Find first non-conflicting single-segment core as chain start.
        // Exclude the last segment so at least one segment remains after start.
        std::optional<std::size_t> chain_start_idx;
        for (std::size_t idx = 0; idx + 1 < segments.size(); ++idx) {
            const auto& seg = segments[idx];
            if (seg.core_idx >= core_work.size()) {
                continue;
            }
            const auto& work = core_work[seg.core_idx];
            if (seg.head_work_index >= work.head_work.size()) {
                continue;
            }
            if (core_chain_info[seg.core_idx].participates) {
                continue;
            }
            if (work.head_work.size() == 1) {
                chain_start_idx = idx;
                break;
            }
        }
        if (!chain_start_idx.has_value()) {
            chains_skipped++;
            continue;
        }

        const std::size_t start = chain_start_idx.value();

        // Build chain from start forward. When allow_wrap_back is true, wrap
        // past the end back to segments before start (used by classic SDPA where
        // q_iter_local is not inflated by prior-head work). When false, only
        // visit segments from start to end (used by ring-joint SDPA to avoid
        // pulling in straddling cores whose q_iter_local is inflated).
        std::vector<std::size_t> chain_order;
        const std::size_t num_steps = allow_wrap_back ? segments.size() : (segments.size() - start);
        for (std::size_t step = 0; step < num_steps; ++step) {
            std::size_t idx = (start + step) % segments.size();
            const auto& seg = segments[idx];
            const uint32_t core_idx = seg.core_idx;
            if (core_idx >= core_work.size() || seg.head_work_index >= core_work[core_idx].head_work.size()) {
                continue;
            }
            if (core_chain_info[core_idx].participates) {
                break;
            }
            chain_order.push_back(idx);
        }

        if (chain_order.size() < 2) {
            chains_skipped++;
            continue;
        }

        // Check if all chain cores have the same q_chunk_count.
        // Mixed q_chunk_count chains are safe in unicast mode when sorted in
        // descending q_chunk_count order: the kernel's should_forward condition
        // guards on (q_iter < next_core_q_chunks), so a heavier sender only
        // forwards for the lighter receiver's iteration count, and the receiver
        // receives for all of its own iterations.  Mcast mode requires uniform
        // q_chunk_count (checked separately in the mcast eligibility pass).
        const uint32_t ref_q = core_work[segments[chain_order[0]].core_idx]
                                   .head_work[segments[chain_order[0]].head_work_index]
                                   .q_chunk_count;
        bool uniform_q = true;
        for (std::size_t i = 1; i < chain_order.size(); ++i) {
            const auto& seg = segments[chain_order[i]];
            if (core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count != ref_q) {
                uniform_q = false;
                break;
            }
        }

        if (uniform_q) {
            // All cores have equal q_chunk_count — safe to pick any injector.
            // Choose the core whose physical X is furthest from existing
            // injectors to spread DRAM reads across channels.
            std::size_t best_pos = 0;
            uint32_t best_dist = 0;
            for (std::size_t pos = 0; pos < chain_order.size(); ++pos) {
                const uint32_t phys_x = core_work[segments[chain_order[pos]].core_idx].physical_core.x;
                uint32_t min_dist = UINT32_MAX;
                for (uint32_t ix : injector_phys_x) {
                    uint32_t d = (phys_x > ix) ? (phys_x - ix) : (ix - phys_x);
                    min_dist = std::min(min_dist, d);
                }
                if (min_dist > best_dist) {
                    best_dist = min_dist;
                    best_pos = pos;
                }
            }
            if (best_pos != 0) {
                std::swap(chain_order[0], chain_order[best_pos]);
            }
        } else {
            // Mixed q_chunk_counts — sort descending so heavier cores come first.
            // Each sender forwards only for min(own_q_iters, next_core_q_chunks)
            // iterations, so a heavier sender safely serves a lighter receiver.
            // Stable sort preserves physical topology where q_counts are equal.
            std::stable_sort(chain_order.begin(), chain_order.end(), [&](std::size_t a, std::size_t b) {
                const auto& seg_a = segments[a];
                const auto& seg_b = segments[b];
                return core_work[seg_a.core_idx].head_work[seg_a.head_work_index].q_chunk_count >
                       core_work[seg_b.core_idx].head_work[seg_b.head_work_index].q_chunk_count;
            });
        }

        const auto& inj_seg = segments[chain_order[0]];
        injector_phys_x.push_back(core_work[inj_seg.core_idx].physical_core.x);
        uint32_t batch = core_work[inj_seg.core_idx].head_work[inj_seg.head_work_index].batch;
        uint32_t head = head_id % num_heads_per_batch;

        log_debug(
            tt::LogOp,
            "Building chain for head {} (batch={}, head={}): {} cores, uniform_q={}, injector phys_x={}",
            head_id,
            batch,
            head,
            chain_order.size(),
            uniform_q,
            core_work[inj_seg.core_idx].physical_core.x);

        for (std::size_t pos = 0; pos < chain_order.size(); ++pos) {
            const std::size_t idx = chain_order[pos];
            const auto& seg = segments[idx];
            const uint32_t core_idx = seg.core_idx;
            const auto& hw = core_work[core_idx].head_work[seg.head_work_index];
            auto& chain = core_chain_info[core_idx];

            chain.participates = true;
            chain.batch = hw.batch;
            chain.head = hw.head;
            chain.q_chunk_start = hw.q_chunk_start;
            chain.q_chunk_count = hw.q_chunk_count;

            if (pos == 0) {
                chain.is_injector = true;
            }
            if (pos == chain_order.size() - 1) {
                chain.is_sink = true;
            }

            // Set prev core coordinates (previous in wrap order)
            if (pos > 0) {
                const uint32_t prev_core_idx = segments[chain_order[pos - 1]].core_idx;
                if (prev_core_idx < core_work.size()) {
                    chain.prev_physical = core_work[prev_core_idx].physical_core;
                }
            }

            // Set next core coordinates and q_chunk count (next in wrap order)
            if (pos + 1 < chain_order.size()) {
                const std::size_t next_idx = chain_order[pos + 1];
                const uint32_t next_core_idx = segments[next_idx].core_idx;
                if (next_core_idx < core_work.size() &&
                    segments[next_idx].head_work_index < core_work[next_core_idx].head_work.size()) {
                    chain.next_physical = core_work[next_core_idx].physical_core;
                    const auto& next_hw = core_work[next_core_idx].head_work[segments[next_idx].head_work_index];
                    chain.next_core_q_chunks = next_hw.q_chunk_count;
                }
            }

            log_debug(
                tt::LogOp,
                "  Core {} in chain: injector={}, sink={}, q_chunks={}, prev={}, next={}",
                core_idx,
                chain.is_injector,
                chain.is_sink,
                chain.q_chunk_count,
                chain.prev_physical,
                chain.next_physical);
        }

        chains_built++;
    }

    log_debug(
        tt::LogOp,
        "Chain construction complete: {} chains built, {} skipped due to conflicts",
        chains_built,
        chains_skipped);
}

// ---- Multicast eligibility checker ----
//
// Checks whether ALL multi-core chains qualify for NOC multicast and,
// if so, configures the injector/receiver fields in core_chain_info.
//
// Returns the number of chains configured for mcast (0 = mcast disabled).
//
// `num_heads_per_batch` is NQH for classic SDPA or NH for ring-joint SDPA.
// It is used solely to decompose head_id -> (batch, head) for the filter
// comparison against CoreChainInfo::batch / ::head.
static inline uint32_t configure_mcast_for_chains(
    const std::vector<std::vector<HeadSegmentRef>>& head_segments,
    std::vector<CoreChainInfo>& core_chain_info,
    const std::vector<CoreWork>& core_work,
    uint32_t num_heads_per_batch) {
    struct McastCandidate {
        std::vector<uint32_t> core_indices;
        uint32_t ref_q_chunks;
    };
    std::vector<McastCandidate> candidates;
    bool all_eligible = true;

    for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
        const auto& segments = head_segments[head_id];
        if (segments.size() < 2) {
            continue;
        }

        // Collect chain core indices that actually participate in this head's chain
        std::vector<uint32_t> chain_core_indices;
        for (const auto& seg : segments) {
            if (seg.core_idx < core_chain_info.size() && core_chain_info[seg.core_idx].participates &&
                core_chain_info[seg.core_idx].batch == (head_id / num_heads_per_batch) &&
                core_chain_info[seg.core_idx].head == (head_id % num_heads_per_batch)) {
                chain_core_indices.push_back(seg.core_idx);
            }
        }

        if (chain_core_indices.size() < 2) {
            continue;
        }

        // Eligibility condition 1: All physical cores share the same Y coordinate
        const uint32_t ref_y = core_work[chain_core_indices[0]].physical_core.y;
        bool same_row = true;
        for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
            if (core_work[chain_core_indices[ci]].physical_core.y != ref_y) {
                same_row = false;
                break;
            }
        }

        if (!same_row) {
            all_eligible = false;
            log_debug(tt::LogOp, "Head {}: mcast ineligible - cores span multiple rows", head_id);
            break;
        }

        // Eligibility condition 2: no non-chain worker cores inside the mcast rectangle.
        uint32_t min_x = core_work[chain_core_indices[0]].physical_core.x;
        uint32_t max_x = min_x;
        for (const auto& ci : chain_core_indices) {
            uint32_t x = core_work[ci].physical_core.x;
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
        }

        bool has_gap = false;
        for (const auto& seg : segments) {
            if (seg.core_idx >= core_work.size()) {
                continue;
            }
            const auto& phys = core_work[seg.core_idx].physical_core;
            if (phys.y == ref_y && phys.x >= min_x && phys.x <= max_x) {
                bool in_chain = false;
                for (const auto& ci : chain_core_indices) {
                    if (ci == seg.core_idx) {
                        in_chain = true;
                        break;
                    }
                }
                if (!in_chain) {
                    has_gap = true;
                    break;
                }
            }
        }

        if (has_gap) {
            all_eligible = false;
            log_debug(tt::LogOp, "Head {}: mcast ineligible - non-chain worker core inside mcast rectangle", head_id);
            break;
        }

        // Eligibility condition 3: All chain cores must have the same q_chunk_count.
        const uint32_t ref_q_chunks = core_chain_info[chain_core_indices[0]].q_chunk_count;
        bool uniform_q_mcast = true;
        for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
            if (core_chain_info[chain_core_indices[ci]].q_chunk_count != ref_q_chunks) {
                uniform_q_mcast = false;
                break;
            }
        }

        if (!uniform_q_mcast) {
            all_eligible = false;
            log_debug(tt::LogOp, "Head {}: mcast ineligible - mixed q_chunk_counts", head_id);
            break;
        }

        // Defensive: crash in all builds if a non-uniform chain slips past the check above.
        for (const auto& ci : chain_core_indices) {
            TT_FATAL(
                core_chain_info[ci].q_chunk_count == ref_q_chunks,
                "Mcast chain for head {} has non-uniform q_chunk_count: core {} has {} vs ref {}",
                head_id,
                ci,
                core_chain_info[ci].q_chunk_count,
                ref_q_chunks);
        }

        candidates.push_back(McastCandidate{std::move(chain_core_indices), ref_q_chunks});
    }

    uint32_t mcast_chains = 0;

    // Only configure mcast if ALL multi-core chains are eligible (all-or-nothing)
    if (all_eligible && !candidates.empty()) {
        mcast_chains = candidates.size();
        for (const auto& cand : candidates) {
            const uint32_t chain_size = cand.core_indices.size();
            const uint32_t num_receivers = chain_size - 1;

            // Find the injector (may not be at index 0 due to rotation)
            uint32_t injector_idx = cand.core_indices[0];
            for (const auto& ci : cand.core_indices) {
                if (core_chain_info[ci].is_injector) {
                    injector_idx = ci;
                    break;
                }
            }

            // Mcast rect covers the full row (min to max physical X across all chain cores).
            uint32_t min_x = core_work[cand.core_indices[0]].physical_core.x;
            uint32_t max_x = min_x;
            for (size_t ci = 1; ci < cand.core_indices.size(); ++ci) {
                uint32_t x = core_work[cand.core_indices[ci]].physical_core.x;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }
            const uint32_t injector_y = core_work[injector_idx].physical_core.y;
            const CoreCoord rect_start = CoreCoord{min_x, injector_y};
            const CoreCoord rect_end = CoreCoord{max_x, injector_y};

            // When the injector is geometrically inside the mcast rect (not at min or max X),
            // the hardware counts it as a destination slot, so num_dests must include it.
            const uint32_t injector_x = core_work[injector_idx].physical_core.x;
            const bool injector_inside_rect = (injector_x > min_x && injector_x < max_x);
            const uint32_t mcast_num_dests = injector_inside_rect ? chain_size : num_receivers;

            // Configure injector
            auto& injector_chain = core_chain_info[injector_idx];
            injector_chain.use_mcast = true;
            injector_chain.prev_physical = rect_start;  // mcast rect start
            injector_chain.next_physical = rect_end;    // mcast rect end
            injector_chain.mcast_num_dests = mcast_num_dests;
            injector_chain.mcast_sender_wait = num_receivers;
            injector_chain.next_core_q_chunks = cand.ref_q_chunks;

            // Configure receivers (all non-injector cores)
            for (const auto& ci : cand.core_indices) {
                if (ci == injector_idx) {
                    continue;
                }
                auto& receiver_chain = core_chain_info[ci];
                receiver_chain.use_mcast = true;
                receiver_chain.prev_physical = core_work[injector_idx].physical_core;
                receiver_chain.next_physical = CoreCoord{0, 0};
                receiver_chain.next_core_q_chunks = 0;
                receiver_chain.is_sink = true;
            }

            log_debug(
                tt::LogOp,
                "Head: mcast enabled - {} receivers, injector core {} (phys_x={}), num_dests={} -> rect ({},{}) to "
                "({},{})",
                num_receivers,
                injector_idx,
                core_work[injector_idx].physical_core.x,
                mcast_num_dests,
                rect_start.x,
                rect_start.y,
                rect_end.x,
                rect_end.y);
        }
    }

    log_info(
        tt::LogOp,
        "Multicast eligibility: {}/{} chains using mcast (all-or-nothing)",
        mcast_chains,
        static_cast<uint32_t>(candidates.size()));

    return mcast_chains;
}

}  // namespace ttnn::prim::detail
