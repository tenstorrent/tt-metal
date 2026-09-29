// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Allocates multicast semaphore IDs in a regular Program through the public host API.

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host_impl.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::kernel_lib::host {
using namespace tt::tt_metal;

void McastImpl::require_program_bound_() const {
    require_arguments_prepared_();
    TT_FATAL(program_bound_, "Call append_semaphores(program) before appending multicast arguments");
}

void McastImpl::require_unbound_() const {
    TT_FATAL(!program_bound_, "A Program-bound Mcast cannot use another attachment or legacy query path");
}

void McastImpl::append_semaphores(Program& program) {
    prepare_arguments_();
    TT_FATAL(!program_bound_, "Multicast semaphores have already been appended");
    std::array<uint32_t, 3> ids{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    for (uint32_t role = 0; role < required_semaphores_(); ++role) {
        ids[role] = CreateSemaphore(program, participating_, 0);
    }
    program_semaphore_ids_ = ids;
    program_bound_ = true;
}

}  // namespace ttnn::kernel_lib::host
