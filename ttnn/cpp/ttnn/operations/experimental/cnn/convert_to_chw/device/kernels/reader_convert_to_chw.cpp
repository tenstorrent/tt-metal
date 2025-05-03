// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    cb_push_back(cb_in, total_tiles);
}
