// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "full_kernel_common.hpp"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t fill_value = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);
    uint32_t num_pages_per_shard_row = get_arg_val<uint32_t>(3);
    uint32_t num_pages_per_shard_col = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    constexpr uint32_t elems_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_width_in_pages = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

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

    const auto dst_accessor = TensorAccessor(dst_args, output_addr, aligned_page_size);

    cb_wait_front(cb_value, 1);

    for (uint32_t shard_row_id = 0; shard_row_id < num_pages_per_shard_col; ++shard_row_id) {
        uint32_t curr_page_id = start_page_id;
        for (uint32_t shard_col_id = 0; shard_col_id < num_pages_per_shard_row; ++shard_col_id) {
            const auto cb_value_addr = get_read_ptr(cb_value);
            noc_async_write_page(curr_page_id, dst_accessor, cb_value_addr);
            curr_page_id++;
        }
        start_page_id += tensor_width_in_pages;
    }

    noc_async_writes_flushed();
    cb_pop_front(cb_value, 1);
    noc_async_write_barrier();
}
