// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_indices_addr = get_arg_val<uint32_t>(1);
    uint32_t start_ht = get_arg_val<uint32_t>(2);
    uint32_t start_wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool src_indices_is_dram = (bool)get_compile_time_arg_val(3);

    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_local = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t tile_bytes_indices = get_tile_size(cb_id_in1);
    constexpr DataFormat data_format_indices = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<src_is_dram> s_input = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast<src_indices_is_dram> s_indices = {
        .bank_base_address = src_indices_addr, .page_size = tile_bytes_indices, .data_format = data_format_indices};

    // Stream in input tensor and indices tensor
    // The input buffer has four tiles as we double-buffer for the two tiles needed for topk_local_sort to start
    // We could load in an entire row of tiles at a time but that would require substantially more memory (we would be
    // double buffering four Wt_local sized CBs)
    for (uint32_t i = start_ht; i < Ht; ++i) {
        for (uint32_t j = start_wt; j < start_wt + Wt_local; ++j) {
            // Read input tensor and indices tensor
            cb_reserve_back(cb_id_in0, onetile);
            cb_reserve_back(cb_id_in1, onetile);
            uint32_t l1_write_addr_input = get_write_ptr(cb_id_in0);
            uint32_t l1_write_addr_indices = get_write_ptr(cb_id_in1);
            noc_async_read_tile(i * Wt + j, s_input, l1_write_addr_input);
            noc_async_read_tile(i * Wt + j, s_indices, l1_write_addr_indices);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            cb_push_back(cb_id_in1, onetile);
        }
    }
}
