// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

#include <algorithm>
#include <tt_stl/assert.hpp>
#include "tt_metal/impl/buffers/semaphore.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::kernel_lib::host {
using namespace tt::tt_metal;

void McastFamily::require_program_bound_() const {
    require_arguments_prepared_();
    TT_FATAL(program_bound_, "Call append_semaphores(program) before appending multicast arguments");
}

void McastFamily::require_unbound_() const {
    TT_FATAL(!program_bound_, "A Program-bound multicast family cannot use another attachment or legacy query path");
}

void McastFamily::validate_semaphores_present_and_zeroed_(
    std::span<const SemaphoreDescriptor> existing, const std::array<uint32_t, 3>& ids) const {
    for (uint32_t role = 0; role < required_semaphores_(); ++role) {
        TT_FATAL(ids[role] < NUM_SEMAPHORES, "No valid multicast semaphore slot available");
        for (uint32_t previous = 0; previous < role; ++previous) {
            TT_FATAL(ids[previous] != ids[role], "Multicast semaphore roles must use distinct IDs");
        }
        auto uncovered = participating_;
        for (const auto& sem : existing) {
            if (sem.core_type == tt::CoreType::WORKER && sem.id == ids[role] &&
                sem.core_ranges.intersects(participating_)) {
                TT_FATAL(sem.initial_value == 0, "Adopted multicast semaphores must start at zero");
                uncovered = uncovered.subtract(sem.core_ranges);
            }
        }
        TT_FATAL(uncovered.empty(), "Adopted multicast semaphore does not cover the participating cores");
    }
}

std::array<uint32_t, 3> McastFamily::resolve_semaphore_ids_(std::span<const SemaphoreDescriptor> existing) const {
    std::array<uint32_t, 3> ids{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    const auto count = required_semaphores_();
    if (cfg_.sem_ids) {
        TT_FATAL(cfg_.sem_ids->size() == count, "Adopt exactly the required multicast semaphore roles");
        std::copy(cfg_.sem_ids->begin(), cfg_.sem_ids->end(), ids.begin());
        validate_semaphores_present_and_zeroed_(existing, ids);
        return ids;
    }
    for (uint32_t role = 0; role < count; ++role) {
        const auto occupied = [&](uint32_t candidate) {
            return std::find(ids.begin(), ids.begin() + role, candidate) != ids.begin() + role ||
                   std::any_of(existing.begin(), existing.end(), [&](const auto& sem) {
                       return sem.core_type == tt::CoreType::WORKER && sem.id == candidate &&
                              sem.core_ranges.intersects(participating_);
                   });
        };
        if (cfg_.base_sem_id) {
            TT_FATAL(*cfg_.base_sem_id < NUM_SEMAPHORES - role, "Multicast semaphore base exceeds available slots");
            ids[role] = *cfg_.base_sem_id + role;
            TT_FATAL(!occupied(ids[role]), "Multicast semaphore ID collides with an existing resource");
        } else {
            for (uint32_t candidate = 0; candidate < NUM_SEMAPHORES; ++candidate) {
                if (!occupied(candidate)) {
                    ids[role] = candidate;
                    break;
                }
            }
            TT_FATAL(ids[role] < NUM_SEMAPHORES, "No valid multicast semaphore slot available");
        }
    }
    return ids;
}

void McastFamily::append_semaphores(Program& program) {
    require_arguments_prepared_();
    TT_FATAL(!program.is_compiled(), "Cannot bind multicast semaphores to a compiled Program");
    TT_FATAL(!program_bound_, "Multicast semaphores have already been appended");

    std::array<uint32_t, 3> ids{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    const auto count = required_semaphores_();
    if (cfg_.sem_ids) {
        // Existing resources belong to the caller; the public Program API does not expose
        // their placement or initial values. Validate the IDs without recreating them.
        TT_FATAL(cfg_.sem_ids->size() == count, "Adopt exactly the required multicast semaphore roles");
        for (uint32_t role = 0; role < count; ++role) {
            ids[role] = (*cfg_.sem_ids)[role];
            TT_FATAL(ids[role] < NUM_SEMAPHORES, "No valid multicast semaphore slot available");
            TT_FATAL(
                std::find(ids.begin(), ids.begin() + role, ids[role]) == ids.begin() + role,
                "Multicast semaphore roles must use distinct IDs");
        }
    } else {
        if (cfg_.base_sem_id) {
            TT_FATAL(*cfg_.base_sem_id <= NUM_SEMAPHORES - count, "Multicast semaphore base exceeds available slots");
        }
        for (uint32_t role = 0; role < count; ++role) {
            ids[role] = CreateSemaphore(program, participating_, 0);
            if (cfg_.base_sem_id) {
                TT_FATAL(
                    ids[role] == *cfg_.base_sem_id + role,
                    "Expected multicast semaphore id {}, got {}",
                    *cfg_.base_sem_id + role,
                    ids[role]);
            }
        }
    }
    program_semaphore_ids_ = ids;
    program_bound_ = true;
}

void Mcast1D::append_semaphores(Program& program) { family_->append_semaphores(program); }
void Mcast2D::append_semaphores(Program& program) { family_->append_semaphores(program); }

}  // namespace ttnn::kernel_lib::host
