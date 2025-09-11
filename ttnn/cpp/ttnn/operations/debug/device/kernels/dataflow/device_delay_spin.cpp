// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// Compile-time arg 0: delay cycles
void kernel_main() {
    constexpr uint32_t delay_cycles = get_compile_time_arg_val(0);
    DPRINT << "Device delay spin kernel started" << ENDL();
    DPRINT << "Delay cycles: " << delay_cycles << ENDL();
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    DPRINT << "Initial wall clock timestamp: " << wall_clock_timestamp << ENDL();
    tt::data_movement::common::spin(delay_cycles);
    uint64_t wall_clock_timestamp_end = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    DPRINT << "Final wall clock timestamp: " << wall_clock_timestamp_end << ENDL();
    DPRINT << "Waited for: " << wall_clock_timestamp_end - wall_clock_timestamp << ENDL() << ENDL();
}
