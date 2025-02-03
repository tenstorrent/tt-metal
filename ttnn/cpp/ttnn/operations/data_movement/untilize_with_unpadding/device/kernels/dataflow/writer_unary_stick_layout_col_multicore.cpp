// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t total_num_rows = get_compile_time_arg_val(3);
    const uint32_t ncores = get_compile_time_arg_val(4);
    const uint32_t third_dim = get_compile_time_arg_val(5);
    const uint32_t tile_width = get_compile_time_arg_val(6);
    /*
    DPRINT << "total_num_rows: " << total_num_rows << ENDL();
    DPRINT << "ncores: " << ncores << ENDL();
    DPRINT << "third dim: " << third_dim <<ENDL();
    DPRINT << "tile_width: " << tile_width <<ENDL();
    */
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t core_number = get_arg_val<uint32_t>(2);

    // DPRINT << "dst_addr: " << dst_addr <<ENDL();
    // DPRINT << "unpadded_X_size: " << unpadded_X_size << ENDL();
    // DPRINT << "core_number: " << core_number <<ENDL();

    constexpr bool dst0_is_dram = get_compile_time_arg_val(0) == 1;

#define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<dst0_is_dram> s = {
        .bank_base_address = dst_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<dst0_is_dram> s = {.bank_base_address = dst_addr, .page_size = unpadded_X_size};
#endif

    auto write_block = [&](uint32_t num_rows,
                           uint32_t mul,
                           uint32_t size_per_row_per_block,
                           uint32_t start_id,
                           uint32_t width_size,
                           uint32_t size_2d) {
        uint32_t onetile = 1;
        bool has_rows = (num_rows) > 0;

        cb_wait_front(cb_id_out0, onetile * has_rows);
        uint32_t l1_read_addr = get_write_ptr(cb_id_out0);
        uint32_t original_addr = get_write_ptr(cb_id_out0);

        // DPRINT << "l1_read_addr: " << l1_read_addr << ENDL();
        // DPRINT << "has_rows: " << uint32_t(has_rows) <<ENDL();

        for (uint32_t k = 0; k < num_rows; k++) {
            // DPRINT << "k: " << k << ENDL();
            uint64_t dst_noc_addr = get_noc_addr(size_2d + k, s);
            // DPRINT << "dst_noc_addr: " << dst_noc_addr <<ENDL();

            uint32_t total_size = mul * size_per_row_per_block + start_id + width_size;
            uint32_t padded_size = total_size - unpadded_X_size;
            uint32_t write_size = width_size;
            // DPRINT << "total_size: " << total_size << ENDL();
            // DPRINT << "padded_size: " << padded_size << ENDL();
            // DPRINT << "write_size: " << write_size <<ENDL();
            if (mul == ncores - 1 && padded_size > 0) {
                write_size = width_size - padded_size;
                // DPRINT << "write_size: " << write_size <<ENDL();
            }

            // Read from DRAM to tmp buffer

            // DPRINT << "WRITING AT: " <<  dst_noc_addr + start_id + mul * size_per_row_per_block << ENDL();
            // DPRINT << "WRITING : " << write_size << " BYTES " << ENDL();
            noc_async_write(l1_read_addr, dst_noc_addr + start_id + mul * size_per_row_per_block, write_size);

            // Block before copying data from tmp to cb buffer
            noc_async_write_barrier();

            // pushing one tile at a time because the current LLK tilize implementation doesn't support tilizing more
            // than one tile per column at the same time this needs to be fixed
            if (k > 0 && (k % tile_width == 0)) {
                // DPRINT << "K >0 && K %32 ==0" <<ENDL();
                // auto* ptr0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
                // for (uint32_t i0 = 0; i0 < 1024; i0 = i0+1) {
                //     DPRINT << "IN THE WRITER VALUE AT i0 = " << (uint32_t)i0 <<  " is: " << BF16((uint16_t)ptr0[i0])
                //     << ENDL();
                // }
                cb_pop_front(cb_id_out0, onetile * has_rows);
                cb_wait_front(cb_id_out0, onetile * has_rows);
                // original_addr = l1_read_addr + width_size;
            }
            // increment by the unpadded size only
            l1_read_addr += width_size;
        }

        // DPRINT << "before last pop" <<ENDL();
        // auto* ptr0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
        // for (uint32_t i0 = 0; i0 < 1024; i0 = i0+1) {
        //     DPRINT << "IN THE WRITER VALUE AT i0 = " << (uint32_t)i0 <<  " is: " << BF16((uint16_t)ptr0[i0])
        //     << ENDL();
        // }
        cb_pop_front(cb_id_out0, onetile * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(3);
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(4);
    const uint32_t width_size = get_arg_val<uint32_t>(5);
    // DPRINT << "size_per_row_per_block: " << size_per_row_per_block << ENDL();
    // DPRINT << "blocks_per_core: " << blocks_per_core << ENDL();
    // DPRINT << "width_size: " << width_size <<ENDL();

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        // DPRINT << "FOR DIM = :" << dim3 <<ENDL();
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            // DPRINT << "for block b =" << b << ENDL();
            write_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
            // DPRINT << "start id: " << start_id <<ENDL();
        }
        size_2d += total_num_rows;
        // DPRINT << "size_2d: " << size_2d <<ENDL();
    }
}
