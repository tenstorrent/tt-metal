// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

TT_KERNEL void writer(uint32_t num_units) {
    // The reader writes directly into the borrowed output shard. This writer only completes the
    // producer/consumer handshake, matching the legacy shared sharded writer.
    DataflowBuffer output(dfb::output);
    output.wait_front(num_units);
    output.pop_front(num_units);
}
