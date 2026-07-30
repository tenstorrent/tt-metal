// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sharded reader: data is already in L1 (CB backed by shard buffer).
// Just push the pages to make them visible to compute.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    CircularBuffer cb_in_obj(cb_in);
    cb_in_obj.push_back(num_pages);
}
