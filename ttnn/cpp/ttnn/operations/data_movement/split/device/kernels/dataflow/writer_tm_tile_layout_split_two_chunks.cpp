// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>

#include "dataflow_api.h"
#include "tensix_types.h"

// #define DEBUG

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(0);
    uint32_t out0_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t out1_tensor_addr = get_arg_val<uint32_t>(2);
    bool out0_only = (bool)get_arg_val<uint32_t>(3);
    bool out1_only = (bool)get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram = get_compile_time_arg_val(1);
    // WRITER COMPILE TIME ARGS
    // constexpr uint32_t out_num_tiles_per_tensor = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_tiles_per_tensor_y = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_tiles_per_tensor_x = get_compile_time_arg_val(3);
    constexpr uint32_t z = get_compile_time_arg_val(4);
    constexpr uint32_t z_stride = get_compile_time_arg_val(5);
    constexpr uint32_t y_stride = get_compile_time_arg_val(6);

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
    constexpr uint32_t onetile = 1;

    constexpr bool tile_dtype_is_bfloat16 = get_compile_time_arg_val(0) == 1;
    constexpr DataFormat data_format = (tile_dtype_is_bfloat16) ? DataFormat::Float16 : DataFormat::Bfp8_b;

    const InterleavedAddrGenFast<out_is_dram_bool> s0 = {
        .bank_base_address = out0_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<out_is_dram_bool> s1 = {
        .bank_base_address = out1_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    std::array<InterleavedAddrGenFast<out_is_dram_bool>, 2> output_banks{s0, s1};
    uint32_t out_split_tensor_tile_id;
    //    uint32_t out_num_tiles_read = out_num_tiles_per_tensor;

    uint32_t bank_id = 0;
    uint32_t tile_id = 0;
#ifdef DEBUG
    // DPRINT << "Writer Tile ID Offset: " << out_tensor_tile_id << ENDL() << ENDL();
#endif
    for (const auto& s : output_banks) {
        if (out1_only && (bank_id == 0)) {
#ifdef DEBUG
            // DPRINT << "Writer is K Only " << ENDL();
#endif
            bank_id++;
            continue;
        }
        uint32_t z_stride_cum = 0;
        for (uint32_t k = 0; k < z; k++) {
            uint32_t y_stride_cum = 0;
            for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
                for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                    uint32_t tile_id = y_stride_cum + z_stride_cum + i;
                    cb_wait_front(cb_id_out0, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    noc_async_write_tile(tile_id + out_tensor_tile_id, s, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, onetile);
#ifdef DEBUG
                    // DPRINT << "Writer for Bank: " << bank_id << " has Tile ID: " << tile_id + out_tensor_tile_id <<
                    // ENDL(); DPRINT << "Writer Address: " << l1_read_addr << ENDL() << ENDL();
#endif
                }
                y_stride_cum += y_stride;
            }
            z_stride_cum += z_stride;
        }
        bank_id++;
        if (out0_only) {
#ifdef DEBUG
            // DPRINT << "Writer is Q Only " << ENDL();
#endif
            break;
        }
    }

#ifdef DEBUG
    // DPRINT << "Writer End " << ENDL();
#endif
}
