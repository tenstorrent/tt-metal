// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// toy_scaled_add writer for a height-sharded output (BRISC).
//
// The output circular buffer is backed by this core's output shard, so compute packs straight into
// the result. This kernel only waits until the whole shard is packed, which keeps the program from
// completing early.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"

using namespace toy_scaled_add;

void kernel_main() {
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    const uint32_t num_rows = get_arg_val<uint32_t>(core_arg::NUM_ROWS);

    CircularBuffer cb_out(cb::OUT);
    cb_out.wait_front(num_rows * Wt);
}
