// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/debug.hpp"

#define DEBUG 1

#ifdef DEBUG
#include "debug/dprint.h"
#endif

#define u16_l1_ptr volatile tt_l1_ptr uint16_t*
#define u32_l1_ptr volatile tt_l1_ptr uint32_t*
#define u8_l1_ptr volatile tt_l1_ptr uint8_t*

#define u8_ptr uint8_t*

template <uint32_t padding_value_num_bytes>
inline __attribute__((always_inline)) void fill_cb_with_padding_value(
    const uint32_t cb, const uint32_t num_bytes, const uint32_t padding_value_as_u32) {
    uint32_t cb_write_addr = get_write_ptr(cb);

    if constexpr (padding_value_num_bytes == 4) {
        u32_l1_ptr cb_write_addr_as_u32 = reinterpret_cast<u32_l1_ptr>(cb_write_addr);
        for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); i++) {
            cb_write_addr_as_u32[i] = padding_value_as_u32;
        }
    } else if constexpr (padding_value_num_bytes == 2) {
        uint16_t padding_value_as_u16 = static_cast<uint16_t>(padding_value_as_u32);
        u16_l1_ptr cb_write_addr_as_u16 = reinterpret_cast<u16_l1_ptr>(cb_write_addr);
        for (uint32_t i = 0; i < num_bytes / sizeof(uint16_t); i++) {
            cb_write_addr_as_u16[i] = padding_value_as_u16;
        }
    } else if constexpr (padding_value_num_bytes == 1) {
        // FIXME: is this actually what we should do for bf8 variants?
        uint8_t padding_value_as_u8 = static_cast<uint8_t>(padding_value_as_u32);
        u8_l1_ptr cb_write_addr_as_u8 = reinterpret_cast<u8_l1_ptr>(cb_write_addr);
        for (uint32_t i = 0; i < num_bytes; i++) {
            cb_write_addr_as_u8[i] = padding_value_as_u8;
        }
    } else {
        static_assert(
            padding_value_num_bytes == 1 || padding_value_num_bytes == 2 || padding_value_num_bytes == 4,
            "padding_value_num_bytes is not 1, 2, or 4");
    }
}

void kernel_main() {
    DPRINT << "entered writer" << ENDL();
    constexpr uint32_t padded_stick_bytes         = get_compile_time_arg_val(0);
    constexpr uint32_t padded_shard_height        = get_compile_time_arg_val(1);
    constexpr uint32_t padding_value_as_u32         = get_compile_time_arg_val(2);
    constexpr uint32_t padding_value_num_bytes      = get_compile_time_arg_val(3);

    constexpr auto input_shard_cb = get_compile_time_arg_val(4);
    constexpr auto output_shard_cb = get_compile_time_arg_val(5);
    constexpr auto padding_value_cb = get_compile_time_arg_val(6);

    cb_reserve_back(output_shard_cb, padded_shard_height); // each page is a padded stick
    uint32_t input_shard_base_addr = get_write_ptr(input_shard_cb);
    uint32_t output_shard_base_addr = get_write_ptr(output_shard_cb);

    fill_cb_with_padding_value<padding_value_num_bytes>(padding_value_cb, padded_stick_bytes, padding_value_as_u32);
    uint32_t padding_value_base_addr = get_read_ptr(padding_value_cb);

    DPRINT << "padding value: " << F32(padding_value_as_u32) << ENDL();

    DPRINT << "padding_value_base_addr: " << padding_value_base_addr << ENDL();
    DPRINT << "padding value f32s: " << ENDL();
    tt::data_movement::common::print_f32_pages(
        padding_value_base_addr, padded_stick_bytes / padding_value_num_bytes, 1);
    DPRINT << ENDL();

    DPRINT << "filled padding value cb" << ENDL();

    for (uint32_t h = 0; h < padded_shard_height; h++) {
        uint32_t output_stick_addr = output_shard_base_addr + h * padded_stick_bytes;
        auto output_stick_ptr = reinterpret_cast<u8_ptr>(output_stick_addr - 16);
        auto padding_value_ptr = reinterpret_cast<u8_ptr>(padding_value_base_addr);
        DPRINT << "writing padding val to stick " << h << ENDL();
        DPRINT << "output_stick_addr: " << output_stick_addr << ENDL();
        DPRINT << "padding_value_base_addr: " << padding_value_base_addr << ENDL();
        DPRINT << "padded_stick_bytes: " << padded_stick_bytes << ENDL();

        DPRINT << "output stick f32s before memcpy: " << ENDL();
        tt::data_movement::common::print_f32_pages(output_stick_addr, padded_stick_bytes / padding_value_num_bytes, 1);
        DPRINT << ENDL();

        DPRINT << "output stick u8s before memcpy: " << ENDL();
        tt::data_movement::common::print_u8_pages(output_stick_addr, padded_stick_bytes, 1);
        DPRINT << ENDL();

        for (uint32_t i = 0; i < padded_stick_bytes; i++) {
            output_stick_ptr[i] = padding_value_ptr[i];
        }
        DPRINT << "wrote padding val to stick " << h << ENDL();

        DPRINT << "output stick f32s after memcpy: " << ENDL();
        tt::data_movement::common::print_f32_pages(output_stick_addr, padded_stick_bytes / padding_value_num_bytes, 1);
        DPRINT << ENDL();

        DPRINT << "output stick u8s after memcpy: " << ENDL();
        tt::data_movement::common::print_u8_pages(output_stick_addr, padded_stick_bytes, 1);
        DPRINT << ENDL();

        cb_push_back(output_shard_cb, 1);
    }
}
