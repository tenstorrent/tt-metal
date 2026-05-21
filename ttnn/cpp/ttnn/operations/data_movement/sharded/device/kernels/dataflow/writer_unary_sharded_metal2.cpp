// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded.cpp.
// Kept side-by-side with the legacy copy during the bulk-port window.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::out_dfb);

    cb_out.wait_front(num_units);
}
