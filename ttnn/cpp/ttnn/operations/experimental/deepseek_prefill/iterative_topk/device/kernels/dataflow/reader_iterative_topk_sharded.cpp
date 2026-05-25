// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sharded reader stub for iterative_topk.
// Input data is already in L1 via the globally-allocated CB.
// This kernel just activates the pages so the writer can consume them.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);

    cb_push_back(cb_input, num_rows);
}
