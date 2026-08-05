// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded.cpp. The legacy file next to this one still serves every op
// that has not migrated to the Metal 2.0 host API; keep the two in sync until the last legacy
// consumer is gone and this file takes over the original's name.
//
// Binding contract a Metal 2.0 factory must supply:
//   dfb::out    CONSUMER binding of the in-place sharded output buffer
//   RTAs        num_units
// There are no compile-time args and no tensor binding: the legacy `dfb_id_out` CTA is now the
// dfb::out binding, and the output is written in place by the producer.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer dfb_out(dfb::out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the buffer balanced.
    dfb_out.pop_front(num_units);
}
