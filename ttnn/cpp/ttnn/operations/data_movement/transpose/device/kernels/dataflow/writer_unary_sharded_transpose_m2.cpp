// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp,
// forked into the transpose op directory because the orchestration constraint forbids editing
// the shared donor in place. Used by the WH-Sharded transpose factory (consumes the borrowed
// output DFB). Sunset when the shared donor and all its consumers migrate to Metal 2.0.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer dfb_out(dfb::out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the DFB balanced.
    dfb_out.pop_front(num_units);
}
