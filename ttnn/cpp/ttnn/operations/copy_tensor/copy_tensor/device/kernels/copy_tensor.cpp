// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t dst_cb_index = get_compile_time_arg_val(2);

    const uint32_t src_addr = get_read_ptr(src_cb_index);
    const uint64_t dst_addr = get_noc_addr(get_write_ptr(dst_cb_index));

    // Copy data
    noc_async_write(src_addr, dst_addr, size_bytes);
    noc_async_write_barrier();
}
