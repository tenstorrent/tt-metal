// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_sharded.cpp, living beside it. The legacy source stays in place,
// non-Metal-2.0, for its ~17 unmigrated binders; ops ported to Metal 2.0 bind this fork.
// Bindings are named for the kernel's role (dfb::out, args::num_units), not for any one consumer.
// Sunset when the last legacy consumer migrates and this fork takes over the legacy name.

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
