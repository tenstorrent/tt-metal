// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

TT_KERNEL void reader(uint32_t num_units) {
    // The input tensor owns the borrowed DFB storage. This shell only publishes
    // readiness to the writer, matching the shared legacy unary reader without
    // migrating or changing its other consumers.
    DataflowBuffer input(dfb::input);
    input.reserve_back(num_units);
    input.push_back(num_units);
}
