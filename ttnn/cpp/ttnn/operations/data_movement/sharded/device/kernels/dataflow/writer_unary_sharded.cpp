// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it as writer_unary_sharded_metal2.cpp. Ops whose
// program factory has been ported to the Metal 2.0 host API bind that file; ops still on the legacy
// host API bind this one. Keep the two in sync until the last legacy consumer is ported, at which
// point this copy can be deleted.

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
