// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "dataflow_api.h"
#include <cstdint>

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t current_device_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    volatile tt_l1_ptr uint32_t* global_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

    noc_semaphore_wait(global_semaphore_addr_ptr, ring_size);
    noc_semaphore_set(global_semaphore_addr_ptr, 0);
}
