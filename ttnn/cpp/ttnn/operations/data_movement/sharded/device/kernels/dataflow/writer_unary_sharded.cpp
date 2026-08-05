// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t dfb_id_out = get_compile_time_arg_val(0);

    DataflowBuffer dfb_out(dfb_id_out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the CB balanced.
    dfb_out.pop_front(num_units);
}
