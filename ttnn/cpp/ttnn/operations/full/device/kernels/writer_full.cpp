// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "full_kernel_common.hpp"

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

    Noc noc;
    CircularBuffer cb(cb_value);

    cb.reserve_back(onepage);

    uint32_t write_addr = cb.get_write_ptr();

    if (val.u == 0) {
        zero_buffer(cb_value, page_size);
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

    cb.push_back(1);

    const auto s = TensorAccessor(dst_args, output_addr);

    cb.wait_front(1);

    uint32_t end_id = start_id + num_pages_per_core;
    for (std::uint32_t i = start_id; i < end_id; i++) {
        noc.async_write(cb, s, s.get_aligned_page_size(), {}, {.page_id = i});
    }
    noc.async_writes_flushed();
    cb.pop_front(1);
    noc.async_write_barrier();
}
