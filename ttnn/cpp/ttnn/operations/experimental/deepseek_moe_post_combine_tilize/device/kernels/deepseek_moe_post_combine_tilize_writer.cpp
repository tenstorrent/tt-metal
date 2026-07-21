// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_tilize_output_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    CircularBuffer cb_tilize_output(cb_tilize_output_id);

    cb_tilize_output.wait_front(num_tiles);
    cb_tilize_output.pop_front(num_tiles);
}
