// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

union value {
    float f;
    uint32_t u;
};
constexpr uint32_t onepage = 1;

void zero_buffer(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
    noc_async_read_barrier();
}

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t fill_value = get_arg_val<uint32_t>(1);
    uint32_t num_pages_per_core = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    constexpr uint32_t elems_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    value val;
    val.u = fill_value;

    cb_reserve_back(cb_value, onepage);

    uint32_t write_addr = get_write_ptr(cb_value);

    if (val.u == 0) {
        zero_buffer(write_addr, page_size);
    } else {
#ifdef OUTPUT_DTYPE_BFLOAT16
        auto ptr = reinterpret_cast<uint16_t*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = val.u >> 16;
        }
#endif
#ifdef OUTPUT_DTYPE_INT32
        auto ptr = reinterpret_cast<uint32_t*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = fill_value;
        }
#endif
#ifdef OUTPUT_DTYPE_FLOAT32
        auto ptr = reinterpret_cast<float*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = val.f;
        }
#endif
    }

    cb_push_back(cb_value, 1);

    const auto s = TensorAccessor(dst_args, output_addr, page_size);

    cb_wait_front(cb_value, 1);

    uint32_t end_id = start_id + num_pages_per_core;
    for (std::uint32_t i = start_id; i < end_id; i++) {
        const auto cb_value_addr = get_read_ptr(cb_value);
        noc_async_write_page(i, s, cb_value_addr);
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_value, 1);
    noc_async_write_barrier();
}
