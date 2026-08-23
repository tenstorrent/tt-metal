// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

TT_KERNEL void reader(uint32_t tiles_per_core) {
    DataflowBuffer input(dfb::input);
    input.push_back(tiles_per_core);
}
