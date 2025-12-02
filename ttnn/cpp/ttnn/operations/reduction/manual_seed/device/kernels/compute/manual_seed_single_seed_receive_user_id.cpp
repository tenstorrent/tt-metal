// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rand.h"
#include "ckernel.h"
#include "ckernel_defs.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t seed = get_compile_time_arg_val(0);

    // Read core ID from mailbox
    bool is_core_id = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);

    // Initialize random generator if core_id matched
    if (is_core_id) {
        rand_tile_init(seed);
    }
}
}  // namespace NAMESPACE
