// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/constants.hpp>
#include "dataflow_api.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t intermed1_cb_id = get_compile_time_arg_val(2);
    constexpr bool output_is_dram = get_compile_time_arg_val(3) == 1;

    auto out_addr = get_arg_val<uint32_t>(0);
    auto start_id = get_arg_val<uint32_t>(1);
    auto num_tiles = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + num_tiles;

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = out_addr,
        .page_size = get_tile_size(intermed1_cb_id),
        .data_format = get_dataformat(intermed1_cb_id)};

    cb_reserve_back(intermed1_cb_id, 1);
    uint32_t intermed1_cb_write_ptr = get_write_ptr(intermed1_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(in_cb_id, 1);
        cb_wait_front(intermed_cb_id, 1);

        uint32_t intermed_cb_read_ptr = get_read_ptr(intermed_cb_id);
        uint32_t in_cb_read_ptr = get_read_ptr(in_cb_id);

        auto in_cb_addr = reinterpret_cast<uint8_t*>(in_cb_read_ptr);
        auto intermed_cb_addr = reinterpret_cast<float*>(intermed_cb_read_ptr);
        auto intermed1_cb_addr = reinterpret_cast<uint8_t*>(intermed1_cb_write_ptr);

        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_cb_addr;

                float input = 0;
#ifdef INPUT_DTYPE_FLOAT32
                input = *reinterpret_cast<float*>(in_cb_addr);
                in_cb_addr += 4;
#endif
#ifdef INPUT_DTYPE_BFLOAT16  // cast: uint16 => uint32 => float and write to input variable.
                uint16_t* in_u16_ptr = reinterpret_cast<uint16_t*>(in_cb_addr);
                uint32_t u32 = static_cast<uint32_t>(*in_u16_ptr) << 16;
                float* f_ptr = reinterpret_cast<float*>(&u32);
                input = *f_ptr;
                in_cb_addr += 2;
#endif
                float output = 0;
                if (rand_float <= input) {
                    output = 1;
                }

#ifdef OUTPUT_DTYPE_FLOAT32
                *(float*)intermed1_cb_addr = output;
                intermed1_cb_addr += 4;
#endif
#ifdef OUTPUT_DTYPE_BFLOAT16
                uint16_t* out_u16_ptr = reinterpret_cast<uint16_t*>(&output) + 1;
                *(uint16_t*)intermed1_cb_addr = *out_u16_ptr;
                intermed1_cb_addr += 2;
#endif
                intermed_cb_addr += 1;
            }
        }
        cb_pop_front(in_cb_id, 1);
        cb_pop_front(intermed_cb_id, 1);

        noc_async_write_tile(i, output_addrg, intermed1_cb_write_ptr);
        noc_async_write_barrier();
    }
}
