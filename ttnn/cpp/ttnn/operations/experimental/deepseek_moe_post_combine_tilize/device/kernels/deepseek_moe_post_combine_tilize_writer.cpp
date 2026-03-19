// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    cb_wait_front(tilize_output_cb_id, num_tiles);
    cb_pop_front(tilize_output_cb_id, num_tiles);
}
