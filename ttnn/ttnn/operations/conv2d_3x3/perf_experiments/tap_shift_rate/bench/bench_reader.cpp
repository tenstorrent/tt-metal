// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Bench reader: the ring CB is backed by the resident slab shard; publish all of its pages once.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t pages = get_compile_time_arg_val(1);
    cb_reserve_back(cb_in, pages);
    cb_push_back(cb_in, pages);
}
