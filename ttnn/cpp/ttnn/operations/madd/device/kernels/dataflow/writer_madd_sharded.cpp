// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

/*
 * Sharded writer for madd operation.
 * Output data stays in L1 (sharded), so we just wait for compute to finish.
 */
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output_index = get_compile_time_arg_val(0);

    // Wait for compute to finish writing all output tiles
    cb_wait_front(cb_output_index, num_tiles);
}
