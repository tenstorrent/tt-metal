// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t total_tiles = get_arg(args::total_tiles);
    DataflowBuffer dfb(dfb::in);
    dfb.push_back(total_tiles);
}
