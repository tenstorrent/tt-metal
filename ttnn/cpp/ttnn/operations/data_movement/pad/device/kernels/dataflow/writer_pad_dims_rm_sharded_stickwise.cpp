// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "dataflow_api.h"

#define u16_l1_ptr volatile tt_l1_ptr uint16_t*
#define u32_l1_ptr volatile tt_l1_ptr uint32_t*

template <uint32_t padding_value_num_bytes, uint32_t num_bytes>
inline __attribute__((always_inline)) void fill_cb_with_padding_value(
    const uint32_t cb, const uint32_t padding_value_as_u32) {
    constexpr uint32_t num_elts =
        num_bytes / padding_value_num_bytes;  // constexpr so that this division happens once on host
    uint32_t cb_write_addr = get_write_ptr(cb);

    if constexpr (padding_value_num_bytes == 4) {
        u32_l1_ptr cb_write_addr_as_u32 = reinterpret_cast<u32_l1_ptr>(cb_write_addr);
        for (uint32_t i = 0; i < num_elts; i++) {
            cb_write_addr_as_u32[i] = padding_value_as_u32;
        }
    } else if constexpr (padding_value_num_bytes == 2) {
        uint16_t padding_value_as_u16 = static_cast<uint16_t>(padding_value_as_u32);
        u16_l1_ptr cb_write_addr_as_u16 = reinterpret_cast<u16_l1_ptr>(cb_write_addr);
        for (uint32_t i = 0; i < num_elts; i++) {
            cb_write_addr_as_u16[i] = padding_value_as_u16;
        }
    } else {
        static_assert(
            padding_value_num_bytes == 2 || padding_value_num_bytes == 4, "padding_value_num_bytes is not 2 or 4");
    }
}

void kernel_main() {
    constexpr uint32_t padded_stick_bytes         = get_compile_time_arg_val(0);
    constexpr uint32_t padded_shard_height        = get_compile_time_arg_val(1);
    constexpr uint32_t padding_value_as_u32         = get_compile_time_arg_val(2);
    constexpr uint32_t padding_value_num_bytes      = get_compile_time_arg_val(3);

    constexpr auto output_shard_cb = get_compile_time_arg_val(4);
    constexpr auto padding_value_cb = get_compile_time_arg_val(5);

    cb_reserve_back(output_shard_cb, padded_shard_height);
    uint32_t output_shard_base_addr = get_write_ptr(output_shard_cb);

    fill_cb_with_padding_value<padding_value_num_bytes, padded_stick_bytes>(padding_value_cb, padding_value_as_u32);
    uint32_t padding_value_base_addr = get_read_ptr(padding_value_cb);

    uint64_t output_stick_noc_addr = get_noc_addr(output_shard_base_addr);
    for (uint32_t h = 0; h < padded_shard_height; h++) {
        noc_async_write(padding_value_base_addr, output_stick_noc_addr, padded_stick_bytes);
        noc_async_write_barrier();

        cb_push_back(output_shard_cb, 1);

        output_stick_noc_addr += padded_stick_bytes;
    }
}
