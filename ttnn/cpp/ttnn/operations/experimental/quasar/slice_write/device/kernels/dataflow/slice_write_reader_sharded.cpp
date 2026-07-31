// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar (Metal-2) slice_write reader. The input is a RESIDENT sharded shard borrowed onto `dfb::in0`,
// so there is no data to fetch — the reader just marks the whole shard available (reserve/push) for the
// writer to drain. Mirrors the shared reader_unary_sharded producer.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_sticks = get_arg(args::num_sticks);
    DataflowBuffer cb_in0(dfb::in0);
    DPRINT("SWR resv n={}\n", num_sticks);  // [#48552 DEBUG] slice_write reader localizer
    cb_in0.reserve_back(num_sticks);
    DPRINT("SWR reserved\n");
    cb_in0.push_back(num_sticks);
    DPRINT("SWR pushed/end\n");
}
