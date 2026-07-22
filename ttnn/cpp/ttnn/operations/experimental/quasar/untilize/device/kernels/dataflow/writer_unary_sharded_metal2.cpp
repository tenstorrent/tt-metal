// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded.cpp. Output is sharded in place (the output DFB is borrowed
// from the output shard buffer), so the data is already where it needs to be; this kernel is only a
// readiness handshake (wait_front then pop_front to keep the DFB balanced). The CB index becomes a
// dfb:: binding and the unit count a named runtime arg. The legacy copy is retained for the
// not-yet-ported sharded untilize factories.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::out);
    cb_out.wait_front(num_units);
    cb_out.pop_front(num_units);
}
