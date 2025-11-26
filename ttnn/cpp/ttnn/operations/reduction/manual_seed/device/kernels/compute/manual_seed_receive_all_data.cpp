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
    constexpr uint32_t user_ids_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t seeds_cb_index = get_compile_time_arg_val(1);

    // Read seed from mailbox
    bool is_core_id = 0;
    UNPACK(is_core_id = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(is_core_id = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(is_core_id = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)

    if (is_core_id) {
        uint32_t seed = 0;
        UNPACK(seed = (uint32_t)mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(seed = (uint32_t)mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(seed = (uint32_t)mailbox_read(ckernel::ThreadId::BriscThreadId);)
        // Set random generator with seed
        rand_tile_init(seed);
    }
}
}  // namespace NAMESPACE
