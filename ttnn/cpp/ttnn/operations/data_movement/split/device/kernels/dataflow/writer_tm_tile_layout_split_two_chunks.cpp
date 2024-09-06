// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include <array>
#include "dataflow_api.h"

#define DEBUG
#ifdef DEBUG
#include "debug/dprint.h"
#endif

void kernel_main() {
    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram = get_compile_time_arg_val(1);
    // WRITER COMPILE TIME ARGS
    //constexpr uint32_t out_num_tiles_per_tensor = get_compile_time_arg_val(2);
    constexpr uint32_t y_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles_per_bank = get_compile_time_arg_val(3);
    constexpr uint32_t z = get_compile_time_arg_val(4);
    constexpr uint32_t z_stride = get_compile_time_arg_val(5);
    constexpr uint32_t y_stride = get_compile_time_arg_val(6);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(7);

    // WRITER RUNTIME ARGS
    std::array<uint32_t, num_chunks> out_addrs;

    for (uint32_t i = 0; i < num_chunks; i++) {
        out_addrs[i] = get_arg_val<uint32_t>(i);
    }

    uint32_t writer_core_id = get_arg_val<uint32_t>(num_chunks+0);
    uint32_t start_tile = get_arg_val<uint32_t>(num_chunks+1);

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
    constexpr uint32_t onetile = 1;

    DataFormat df;

#define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
#if (tile_dtype_is_bfloat16)
    df = DataFormat::Float16;
#else
    df = DataFormat::Bfp8_b;
#endif
    std::array<InterleavedAddrGenFast<out_is_dram_bool>, num_chunks> output_banks;
    std::transform(out_addrs.begin(),
                   out_addrs.end(),
                   output_banks.begin(),
                   [&](uint32_t &addr) -> InterleavedAddrGenFast<out_is_dram_bool> {
                        return { .bank_base_address = addr,
                                 .page_size = single_tile_size_bytes,
                                 .data_format = df };
                   });

    uint32_t out_split_tensor_tile_id;
    uint32_t bank_id = 0;
    uint32_t tile_id = 0;

    for (const auto& s : output_banks) {
        uint32_t z_stride_cum = 0;
        for (uint32_t k = 0; k < z; k++) {
            uint32_t y_stride_cum = 0;
            for (uint32_t j = 0; j < y_tiles_per_core; j++) {
                for (uint32_t i = 0; i < x_tiles_per_bank; i++) {
                    uint32_t tile_id = start_tile + y_stride_cum + z_stride_cum + i;
                    cb_wait_front(cb_id_out0, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    noc_async_write_tile(tile_id, s, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, onetile);
#ifdef DEBUG
                    DPRINT << "[W/" << writer_core_id <<"] B" << bank_id << "@" << s.bank_base_address << "[" << tile_id << "]" << ENDL();
#endif
                }
                y_stride_cum += y_stride;
            }
            z_stride_cum += z_stride;
        }
        bank_id++;
    }
}
