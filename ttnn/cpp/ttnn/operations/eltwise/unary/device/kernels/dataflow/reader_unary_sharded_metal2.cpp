// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of reader_unary_sharded.cpp, which lives beside it. Ops ported to
// Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until the last
// of them migrates and the original is retired, changes here likely belong there too.
//
// The binding name below (dfb::in) and the named argument set are this fork's interface: every later
// consumer inherits them, so they are taken from the kernel's own vocabulary rather than any one op's
// locals, and are not renamed once a consumer exists. The input DFB is a borrowed-memory buffer backed
// by the sharded input tensor; this reader only produces the readiness handshake (push_back).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_tiles_per_core = get_arg(args::num_tiles_per_core);

    DataflowBuffer dfb(dfb::in);
    dfb.push_back(num_tiles_per_core);
}
