// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc_semaphore.h"
#include "experimental/circular_buffer.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr bool rms_norm = get_compile_time_arg_val(15) == 1;

    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    experimental::Semaphore<> reduce_sender_sem(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ex_global_obj(cb_ex_global);

    reduce_sender_sem.set(INVALID);
    cb_ex_global_obj.reserve_back(stats_tiles * block_h);
    reduce_sender_sem.wait(VALID);
    cb_ex_global_obj.push_back(stats_tiles * block_h);
}
