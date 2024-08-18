// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"


// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {

    const uint32_t num_tiles  = get_arg_val<uint32_t>(0);
    const uint32_t start_tensor = get_arg_val<uint32_t>(1);
    const uint32_t start_tensor_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    uint8_t l1_src_addr_gens_memblk[sizeof(InterleavedAddrGenFast<false>) * num_tensors];
    uint8_t dram_src_addr_gens_memblk[sizeof(InterleavedAddrGenFast<true>) * num_tensors];

    InterleavedAddrGenFast<false> *l1_src_addr_gens = reinterpret_cast<InterleavedAddrGenFast<false>*>(l1_src_addr_gens_memblk);
    InterleavedAddrGenFast<true> *dram_src_addr_gens = reinterpret_cast<InterleavedAddrGenFast<true>*>(dram_src_addr_gens_memblk);

    bool is_dram[num_tensors];
    uint32_t num_tiles_per_block[num_tensors];
    uint32_t tile_id_per_tensor[num_tensors];
    constexpr uint32_t src_addr_base_idx = 3;
    constexpr uint32_t is_dram_base_offset = num_tensors;
    constexpr uint32_t num_tiles_per_block_base_offset = is_dram_base_offset + num_tensors;
    constexpr uint32_t tile_id_per_tensor_offset = num_tiles_per_block_base_offset + num_tensors;
    tt_l1_ptr uint32_t * arg_ptr = (tt_l1_ptr uint32_t*) get_arg_addr(src_addr_base_idx);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t src_addr  = arg_ptr[i];
        is_dram[i] = (bool)arg_ptr[is_dram_base_offset + i];
        num_tiles_per_block[i] = arg_ptr[num_tiles_per_block_base_offset + i];
        tile_id_per_tensor[i] = arg_ptr[tile_id_per_tensor_offset + i];
        if (is_dram[i]) {
            new(&dram_src_addr_gens[i]) InterleavedAddrGenFast<true>{
                .bank_base_address = src_addr,
                .page_size = tile_size_bytes,
                .data_format = data_format
            };
        } else {
            new(&l1_src_addr_gens[i]) InterleavedAddrGenFast<false>{
                .bank_base_address = src_addr,
                .page_size = tile_size_bytes,
                .data_format = data_format
            };
        }
    }

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_id_in, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
        if (is_dram[curr_tensor]) {
            noc_async_read_tile(tile_id_per_tensor[curr_tensor], dram_src_addr_gens[curr_tensor], l1_write_addr);
        } else {
            noc_async_read_tile(tile_id_per_tensor[curr_tensor], l1_src_addr_gens[curr_tensor], l1_write_addr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in, ublock_size_tiles);

        tile_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_tiles_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
    }
}
