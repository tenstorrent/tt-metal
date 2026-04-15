// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "experimental/circular_buffer.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    experimental::CircularBuffer cb_in(cb_id_in0);

    cb_in.reserve_back(num_units);
    cb_in.push_back(num_units);
}
