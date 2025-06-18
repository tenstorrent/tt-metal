// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    DPRINT << "START WAIT " << num_units << " " << cb_id_out << ENDL();
    cb_wait_front(cb_id_out, num_units);
    DPRINT << "DONE WRITING " << cb_id_out << ENDL();
}
