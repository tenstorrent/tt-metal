// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// Compile-time arg 0: delay cycles
void kernel_main() {
    constexpr uint32_t delay_cycles = get_compile_time_arg_val(0);
    tt::data_movement::common::spin(delay_cycles);
}
