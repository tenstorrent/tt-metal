// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>

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
            break;
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
        }
    }

    return mcast_chains;
}

}  // namespace ttnn::prim::detail
