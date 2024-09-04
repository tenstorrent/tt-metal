// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#define DEBUG
#ifdef DEBUG
#include "debug/dprint.h"
#endif

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t start_tile = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t reader_core_id = get_arg_val<uint32_t>(2);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t in0_is_dram = get_compile_time_arg_val(1);
    constexpr uint32_t z = get_compile_time_arg_val(2);
    constexpr uint32_t z_stride = get_compile_time_arg_val(3);
    constexpr uint32_t y_stride = get_compile_time_arg_val(4);
    constexpr uint32_t y_tiles_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t x_tiles_per_core = get_compile_time_arg_val(6);

    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;

    constexpr uint32_t onetile = 1;
#define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
#if (tile_dtype_is_bfloat16)
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Float16};
#else
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Bfp8_b};
#endif

#ifdef DEBUG
    DPRINT << "Reader Start Tile: " << start_tile << ENDL() << ENDL();
    DPRINT << "Reader Z Stride: " << z_stride << ENDL();
    DPRINT << "Reader Y Stride: " << y_stride << ENDL();
#endif

    uint32_t z_stride_cum = 0;
    for (uint32_t k = 0; k < z; k++) {
        uint32_t y_stride_cum = 0;
        for (uint32_t j = 0; j < y_tiles_per_core; j++) {
            for (uint32_t i = 0; i < x_tiles_per_core; i++) {
                uint32_t tile_offset = y_stride_cum + z_stride_cum + i;
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(start_tile + tile_offset, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
#ifdef DEBUG
                DPRINT << "Reader Core ID: " << reader_core_id << ENDL();
                DPRINT << "Reader Tile ID: " << start_tile + tile_offset << ENDL();
                DPRINT << "Reader Address: " << l1_write_addr_in0 << ENDL() << ENDL();
#endif
            }
            y_stride_cum += y_stride;
        }
        z_stride_cum += z_stride;
    }
}
