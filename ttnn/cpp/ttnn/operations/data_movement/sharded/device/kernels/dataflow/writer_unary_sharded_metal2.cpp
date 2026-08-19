// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of writer_unary_sharded.cpp, which lives beside it. Ops ported to
// Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until the last
// of them migrates and the original is retired, changes here likely belong there too.
//
// The binding name below (dfb::out) and the named argument set are this fork's interface: every later
// consumer inherits them, so they are taken from the kernel's own vocabulary rather than any one op's
// locals, and are not renamed once a consumer exists. The output DFB is a borrowed-memory buffer backed
// by the sharded output tensor; the data is already in place, so this writer only performs the
// wait/pop readiness handshake to keep the DFB balanced.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_units = get_arg(args::num_units);

    DataflowBuffer dfb_out(dfb::out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the DFB balanced.
    dfb_out.pop_front(num_units);
}
