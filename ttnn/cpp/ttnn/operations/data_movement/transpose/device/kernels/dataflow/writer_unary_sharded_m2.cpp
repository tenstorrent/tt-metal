// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp, for the
// sharded transpose WH writer (the legacy source is shared by ~a dozen ops; not editable in place).
// Behavior unchanged: wait on the borrowed output DFB for num_units entries. The legacy CT (output CB
// index) is now the dfb::out binding.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t cb_id_out = dfb::out;

    DataflowBuffer cb_out(cb_id_out);

    cb_out.wait_front(num_units);
}
