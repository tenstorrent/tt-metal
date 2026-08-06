// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sharded writer: output CB is backed by shard buffer.
// Just wait for compute to fill it — data is already in place.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    CircularBuffer cb_out_obj(cb_out);
    cb_out_obj.wait_front(num_pages);
}
