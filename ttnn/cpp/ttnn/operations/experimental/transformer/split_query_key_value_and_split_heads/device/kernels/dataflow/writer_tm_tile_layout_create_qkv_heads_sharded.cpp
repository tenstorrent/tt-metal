// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    constexpr uint32_t num_heads_per_tensor = get_compile_time_arg_val(0);  // 2
    constexpr uint32_t block_ht = get_compile_time_arg_val(1);              // 12
    constexpr uint32_t block_wt = get_compile_time_arg_val(2);              // 12
    constexpr uint32_t out_block_wt = get_compile_time_arg_val(3);          // 2
    constexpr uint32_t block_wt_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t out_block_wt_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_tensor = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_stride_size_bytes = get_compile_time_arg_val(7);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_im0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;
    constexpr uint32_t cb_out2 = tt::CBIndex::c_18;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_in0);
    const DataFormat data_format = get_dataformat(cb_in0);

    // re-order k
    uint32_t l1_read_addr = get_read_ptr(cb_in0);
    l1_read_addr += tensor_stride_size_bytes;
    uint32_t src_noc_addr_offset_outer_most = 0;
    for (uint32_t j = 0; j < num_heads_per_tensor; j++) {  // 2
        uint32_t src_noc_addr_offset_outer = 0;
        for (uint32_t k = 0; k < out_block_wt; k++) {  // 2
            uint32_t l1_read_addr_offset = 0;
            uint32_t l1_write_addr_out1 = get_write_ptr(cb_im0);
            cb_reserve_back(cb_im0, block_ht);
            for (uint32_t i = 0; i < block_ht; i++) {  // 12
                uint64_t src_noc_addr = get_noc_addr(
                    l1_read_addr + l1_read_addr_offset + src_noc_addr_offset_outer + src_noc_addr_offset_outer_most);
                noc_async_read(src_noc_addr, l1_write_addr_out1, single_tile_size_bytes);
                l1_read_addr_offset += block_wt_size_bytes;
                l1_write_addr_out1 += single_tile_size_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_im0, block_ht);
            src_noc_addr_offset_outer += single_tile_size_bytes;
        }
        src_noc_addr_offset_outer_most += out_block_wt_size_bytes;
    }
}
