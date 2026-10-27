// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local reader for transpose's sharded WH factory. Trivial sharded "reader": the input
// shard already resides in L1 (the cb_in0 DFB borrows the input tensor's shard memory), so there is
// no NoC read — it just publishes the resident tiles to the compute consumer. Only the resource
// bindings move to the Metal 2.0 namespaces (dfb::/args::).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg(args::num_tiles);

    DataflowBuffer cb(dfb::cb_in0);
    cb.push_back(num_tiles_per_core);
}
