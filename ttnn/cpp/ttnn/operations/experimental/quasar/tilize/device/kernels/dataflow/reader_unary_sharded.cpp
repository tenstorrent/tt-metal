// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);

    DataflowBuffer cb(dfb::in);
    cb.push_back(num_tiles_per_core);
}
