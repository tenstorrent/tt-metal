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
    // Read core ID from mailbox
    bool is_core_id = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);

    if (is_core_id) {
        // Read seed from mailbox
        uint32_t seed = (uint32_t)mailbox_read(ckernel::ThreadId::BriscThreadId);

        // Set random generator with seed
        rand_tile_init(seed);
    }
}
}  // namespace NAMESPACE
