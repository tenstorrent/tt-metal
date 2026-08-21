// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_sharded.cpp, living beside it. The legacy source stays in place,
// non-Metal-2.0, for its ~18 unmigrated binders; ops ported to Metal 2.0 bind this fork.
// Bindings are named for the kernel's role (dfb::in, args::num_tiles), not for any one consumer.
// Sunset when the last legacy consumer migrates and this fork takes over the legacy name.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg(args::num_tiles);

    DataflowBuffer dfb(dfb::in);
    dfb.push_back(num_tiles_per_core);
}
