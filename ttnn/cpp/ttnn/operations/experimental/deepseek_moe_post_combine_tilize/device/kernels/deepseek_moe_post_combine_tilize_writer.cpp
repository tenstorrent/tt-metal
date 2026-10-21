// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_tilize_output(dfb::tilize_output);

    dfb_tilize_output.wait_front(num_tiles);
    dfb_tilize_output.pop_front(num_tiles);
}
