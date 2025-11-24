// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    cb_wait_front(output_cb, num_tiles);
}
