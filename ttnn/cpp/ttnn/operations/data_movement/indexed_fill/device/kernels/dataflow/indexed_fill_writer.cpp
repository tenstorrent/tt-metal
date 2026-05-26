// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Native fast-path writer for ttnn.indexed_fill.
//
// The data CB is globally allocated to the output buffer, so the reader writing into the CB
// is the same as writing into the output.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t batch_size_in_pages = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    if (batch_size_in_pages == 0) {
        return;
    }

    cb_wait_front(cb_id_in0, batch_size_in_pages);
    cb_pop_front(cb_id_in0, batch_size_in_pages);
}
