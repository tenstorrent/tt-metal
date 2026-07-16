// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Native fast-path writer for ttnn.indexed_fill.
//
// The data CB is globally allocated to the output buffer, so the reader writing into the CB
// is the same as writing into the output.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t batch_size_in_pages = get_arg_val<uint32_t>(0);

    constexpr uint32_t dfb_in0_id = get_compile_time_arg_val(0);

    if (batch_size_in_pages == 0) {
        return;
    }

    DataflowBuffer dfb_in0(dfb_in0_id);

    dfb_in0.wait_front(batch_size_in_pages);
    dfb_in0.pop_front(batch_size_in_pages);
}
