// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    cb_wait_front(cb_id_out, 2);
}
