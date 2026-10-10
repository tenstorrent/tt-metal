// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::out);

    // Output is sharded in place, so the data is already where it needs to be; the
    // wait below is only a readiness handshake. Pop to leave the CB balanced.
    // Thread t of N owns entries t, t + N, ... spread over several tile counters, and wait/pop
    // move one counter per call, so entries are taken one at a time.
    for (uint32_t k = get_my_thread_id(); k < num_units; k += get_num_threads()) {
        cb_out.wait_front(1);
        cb_out.pop_front(1);
    }
}
