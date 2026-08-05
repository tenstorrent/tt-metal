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
    // [#48552 DIAG - remove after] confirm the borrowed-DFB fake-push count vs the device credit capacity
    // (mirrors the s2i reader diagnostic). If push > cap the push_back asserts; otherwise the tilize hang
    // (0x19 MEM_READ_NO_RESPONSE) is downstream in the compute tilize, not the reader.
    DPRINT("[TLZR] push={} cap={}\n", (uint32_t)num_tiles_per_core, (uint32_t)cb.get_local_num_entries());
    cb.push_back(num_tiles_per_core);
    DPRINT("[TLZR] post-push DONE\n");
}
