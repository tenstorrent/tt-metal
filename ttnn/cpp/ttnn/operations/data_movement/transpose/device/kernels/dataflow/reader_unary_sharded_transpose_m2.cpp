// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp, forked into
// the transpose op directory because the orchestration constraint forbids editing the shared
// donor in place. Used by the WH-Sharded transpose factory (produces the borrowed input DFB).
// Sunset when the shared donor and all its consumers migrate to Metal 2.0.

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
