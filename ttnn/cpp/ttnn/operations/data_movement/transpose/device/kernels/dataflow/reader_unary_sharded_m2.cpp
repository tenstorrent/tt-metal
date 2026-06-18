// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp, for the sharded
// transpose WH reader (the legacy source is shared by ~a dozen ops; not editable in place). Behavior
// unchanged: push the borrowed input DFB by num_tiles_per_core. The legacy CT (src0 CB index) is now
// the dfb::src0 binding.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);
    constexpr uint32_t cb_id_in0 = dfb::src0;

    DataflowBuffer cb(cb_id_in0);
    cb.push_back(num_tiles_per_core);
}
