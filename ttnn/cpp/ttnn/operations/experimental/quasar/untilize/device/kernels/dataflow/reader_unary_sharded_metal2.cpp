// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_sharded.cpp. Even-sharding zero-copy path: the input DFB is built
// directly on the input shard's L1 buffer (DataflowBufferSpec.borrowed_from = input), so this reader
// only advances the DFB write pointer (push_back) — no NoC read, no TensorAccessor. The CB index is
// a dfb:: binding and the per-core tile count is a named runtime arg. The legacy reader_unary_sharded
// is retained for the not-yet-ported sharded untilize factories.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);

    DataflowBuffer cb(dfb::in);
    cb.push_back(num_tiles_per_core);
}
