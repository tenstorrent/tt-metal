// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    CircularBuffer cb_out(cb_id_out);
    cb_out.wait_front(2);
}
