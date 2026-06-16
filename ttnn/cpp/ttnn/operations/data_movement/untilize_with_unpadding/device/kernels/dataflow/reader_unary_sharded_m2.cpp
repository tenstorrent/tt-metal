// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp.
// The legacy source is shared by ~12 ops, so it is forked here (not edited in place) and
// ported to Metal 2.0 named bindings for untilize_with_unpadding's multi-core sharded factory.
// The sharded input lives in L1: the reader merely advances the DFB write pointer (push_back)
// to publish the resident tiles to the compute consumer; there is no NOC read.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);

    DataflowBuffer dfb_in(dfb::in);
    dfb_in.push_back(num_tiles_per_core);
}
