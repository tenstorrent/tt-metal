// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of
// ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp
// Forked (not modified in place) because the legacy source is shared by many ops that have not yet
// migrated to Metal 2.0 named bindings. See METAL2_PORT_REPORT.md.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::out);
    cb_out.wait_front(num_units);
}
