// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <tt_stl/assert.hpp>
#include "kernels/dataflow/chunked_prefill_utils.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct RotatedQLockstepGroup {
    std::vector<uint32_t> members;  // core indices in multicast rectangle order
    uint32_t injector_pos = 0;      // index into members
};

struct RotatedQIteration {
    uint32_t remainder_start = kRotatedNoRemainder;  // first flat chunk of the moving unit
    bool group_has_remainder = false;                // includes padded multicast synchronization slots
    uint32_t float_dest_core = kRotatedNoDest;  // logical core index; factory packs physical coordinates
};

using RotatedQSchedule = std::vector<std::vector<RotatedQIteration>>;  // [core][active ordinal]

// Pure host plan over validated, equal-size multicast groups covering every core once.
// The factory owns eligibility, physical coordinates, semaphore allocation and runtime-arg packing.
// Build all ordinals independently of the current active mask, which can change on cache hits.
inline RotatedQSchedule build_rotated_q_schedule(
    uint32_t num_cores,
    uint32_t ring_size,
    uint32_t rotated_base_chunks,
    uint32_t rotated_float_chunks,
    uint32_t rotation_unit_chunks,
    std::span<const RotatedQLockstepGroup> rotated_groups) {
    TT_ASSERT(rotation_unit_chunks == 1 || rotation_unit_chunks == 2);
    TT_ASSERT(rotated_base_chunks >= rotation_unit_chunks && rotated_base_chunks % rotation_unit_chunks == 0);
    TT_ASSERT(!rotated_groups.empty() && !rotated_groups.front().members.empty());
    const uint32_t num_groups = static_cast<uint32_t>(rotated_groups.size());
    const uint32_t rotated_group_size = static_cast<uint32_t>(rotated_groups.front().members.size());
    const uint32_t groups_needed = (rotated_float_chunks + rotated_group_size - 1) / rotated_group_size;
    // Put the first remainder on the injector so it never runs padded slots.
    // Rotate groups each iteration; each core owns at most one remainder unit.
    // Balanced zigzag units contain both adjacent flat IDs, preserving low/high slot parity.
    auto float_owner = [&](uint32_t ring_iter, uint32_t float_idx) {
        const uint32_t first_group = ring_iter * groups_needed;
        const uint32_t float_group_offset = float_idx / rotated_group_size;
        const uint32_t group_idx = (first_group + float_group_offset) % num_groups;
        const auto& group = rotated_groups[group_idx];
        const uint32_t pos_in_group = float_idx % rotated_group_size;
        const uint32_t injector_pos = group.injector_pos;
        const bool is_injector_slot = pos_in_group == 0;
        const uint32_t member_idx =
            is_injector_slot ? injector_pos : (pos_in_group <= injector_pos ? pos_in_group - 1 : pos_in_group);
        return group.members[member_idx];
    };
    RotatedQSchedule rotated_sched(num_cores, std::vector<RotatedQIteration>(ring_size));
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        for (uint32_t float_idx = 0; float_idx < rotated_float_chunks; ++float_idx) {
            const uint32_t owner = float_owner(ring_iter, float_idx);
            auto& sched = rotated_sched[owner][ring_iter];
            // At most one remainder unit per core; fixed base IDs are derived locally.
            TT_ASSERT(sched.remainder_start == kRotatedNoRemainder);
            sched.remainder_start = rotated_base_chunks * num_cores + float_idx * rotation_unit_chunks;
            if (ring_iter > 0) {
                const uint32_t previous_owner = float_owner(ring_iter - 1, float_idx);
                if (previous_owner != owner) {
                    // The receiver derives migration by comparing consecutive remainder IDs.
                    rotated_sched[previous_owner][ring_iter - 1].float_dest_core = owner;
                }
            }
        }
        // Every core in a group runs the group's max slot count, so padded members still relay
        // the mcast handshakes.
        for (const auto& group : rotated_groups) {
            // The injector owns the first remainder, so its count is the group maximum.
            const bool group_has_remainder =
                rotated_sched[group.members[group.injector_pos]][ring_iter].remainder_start != kRotatedNoRemainder;
            for (const uint32_t ci : group.members) {
                rotated_sched[ci][ring_iter].group_has_remainder = group_has_remainder;
            }
        }
    }
    return rotated_sched;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
