// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

typedef union {
    float f;
    uint32_t u;
} u;
constexpr uint32_t onetile = 1;

void kernel_main() {
    uint32_t fill_value = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    const uint32_t cb_page_size = get_tile_size(cb_value);
    const auto cb_data_format = get_dataformat(cb_value);

    u val;
    val.u = fill_value;

    cb_reserve_back(cb_value, onetile);

    uint32_t write_addr = get_write_ptr(cb_value);

#ifdef OUTPUT_DTYPE_BFLOAT16
    auto ptr = reinterpret_cast<uint16_t *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = val.u >> 16;
    }
#endif
#ifdef OUTPUT_DTYPE_INT32
    auto ptr = reinterpret_cast<uint32_t *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = fill_value;
    }
#endif
#ifdef OUTPUT_DTYPE_FLOAT32
    auto ptr = reinterpret_cast<float *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = val.f;
    }
#endif
    cb_push_back(cb_value, 1);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = output_addr, .page_size = cb_page_size, .data_format = cb_data_format};

    cb_wait_front(cb_value, 1);

    uint32_t end_id = start_id + num_tiles;
    for (std::uint32_t i = start_id; i < end_id; i++) {
        const auto cb_value_addr = get_read_ptr(cb_value);
        noc_async_write_tile(i, s, cb_value_addr);
        noc_async_write_barrier();
    }
    cb_pop_front(cb_value, 1);
}
