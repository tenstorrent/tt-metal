// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local writer for transpose's sharded WH factory. Trivial sharded "writer": the output
// shard is produced in place into the cb_out0 DFB (which borrows the output tensor's shard memory),
// so there is no NoC write — it just waits for the compute producer to finish. Only the resource
// bindings move to the Metal 2.0 namespaces (dfb::/args::).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::cb_out0);
    cb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the CB balanced.
    cb_out.pop_front(num_units);
}
