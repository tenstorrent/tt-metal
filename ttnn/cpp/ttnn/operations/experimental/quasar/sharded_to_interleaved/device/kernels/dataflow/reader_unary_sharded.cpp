// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles_per_core = get_arg(args::num_units);

    // The input shard lives in resident L1; the DFB is borrowed onto the input buffer
    // (DataflowBufferSpec::borrowed_from). The reader does a fake-push so the downstream
    // consumer (writer, or compute on the convert_df path) sees the shard's tiles available.
    DataflowBuffer cb_in0(dfb::in0);
    cb_in0.push_back(num_tiles_per_core);
}
