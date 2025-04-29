// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint16_t wt_offset = wt << 5;  // multiply by 32 to get the starting offset for this tile

    uint32_t count = 0;
    for (uint32_t face_h = 0; face_h < 2; ++face_h) {
        for (uint32_t face_w = 0; face_w < 2; ++face_w) {
            for (uint32_t elem_h = 0; elem_h < 16; ++elem_h) {
                for (uint32_t elem_w = 0; elem_w < 16; elem_w += 2) {
                    uint16_t value = elem_w + 16 * face_w + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }

    // // Write the first row of the tile
    // uint16_t value = wt_offset;
    // for (uint32_t face_w = 0; face_w < 2; ++face_w) {
    //     for (uint32_t elem_w = 0; elem_w < 16; elem_w += 2) {
    //         uint16_t value = value + elem_w;
    //         ptr[count] = (value + 1) << 16 | value;
    //         count++;
    //     }
    // }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t indices_addr = get_arg_val<uint32_t>(1);
    uint32_t start_ht = get_arg_val<uint32_t>(2);
    uint32_t start_wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool indices_is_dram = (bool)get_compile_time_arg_val(3);

    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_local = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);
    constexpr uint32_t tile_bytes_indices = get_tile_size(cb_id_in1);
    constexpr DataFormat data_format_indices = get_dataformat(cb_id_in1);

    DPRINT << "cb_id_in0: " << cb_id_in0 << ENDL();
    DPRINT << "cb_id_in1: " << cb_id_in1 << ENDL();
    DPRINT << "src_is_dram: " << src_is_dram << ENDL();
    DPRINT << "indices_is_dram: " << indices_is_dram << ENDL();
    DPRINT << "Ht: " << Ht << ENDL();
    DPRINT << "Wt_local: " << Wt_local << ENDL();
    DPRINT << "Wt: " << Wt << ENDL();

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast<indices_is_dram> s_indices = {
        .bank_base_address = indices_addr, .page_size = tile_bytes_indices, .data_format = data_format_indices};

    // Stream in input tensor and generate the relevant index tensor tiles
    // The input buffer has four tiles as we double-buffer for the two tiles needed for topk_local_sort to start
    // We could load in an entire row of tiles at a time but that would require substantially more memory (we would be
    // double buffering four Wt_local sized CBs)
    for (uint32_t i = start_ht; i < Ht; ++i) {
        DPRINT << "i: " << i << ENDL();
        for (uint32_t j = start_wt; j < start_wt + Wt_local; ++j) {
            DPRINT << "j: " << j << ENDL();
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(i * Wt + j, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            // generate_index_tile(cb_id_in1, j);  // index tensor tile at width chunk j
            cb_reserve_back(cb_id_in1, onetile);
            uint32_t l1_write_addr_indices = get_write_ptr(cb_id_in1);
            noc_async_read_tile(i * Wt + j, s_indices, l1_write_addr_indices);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onetile);
        }
    }
}
