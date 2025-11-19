// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);  // this si not een beign used ?
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(index++);
    const uint32_t D = get_arg_val<uint32_t>(index++);
    const uint32_t N = get_arg_val<uint32_t>(index++);
    const uint32_t C = get_arg_val<uint32_t>(index++);
    const uint32_t Ht = get_arg_val<uint32_t>(index++);
    const uint32_t Wt = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);  // collapsed dims > 5
    const uint32_t current_row = get_arg_val<uint32_t>(index++);
    const uint32_t num_rows = get_arg_val<uint32_t>(index++);
    uint32_t page_size = get_arg_val<uint32_t>(index++);
    constexpr uint32_t onetile = 1;

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    const uint32_t tile_hw = get_tile_hw(cb_id_dst);  // Number of elements

    const uint32_t aligned_page_size = ((page_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

#if !DST_SHARDED
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const uint32_t element_size = dst_tile_bytes / tile_hw;

    const auto dst = TensorAccessor(dst_args, dst_addr, aligned_page_size);  // use aligned_page_size
#endif

    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;

    uint32_t num_tiles_written = 0;

    auto row_width = aligned_page_size / element_size;
    auto max_elements = row_width > tile_hw ? row_width : tile_hw;
    const uint32_t div = (row_width + tile_hw - 1) / tile_hw;
    uint32_t bytes_to_write = aligned_page_size > dst_tile_bytes ? dst_tile_bytes : aligned_page_size;

    DPRINT << "Writer dst_tile_bytes: " << dst_tile_bytes << ", page_size: " << page_size
           << ", Aligned page_size : " << aligned_page_size << ", div: " << div << ENDL();

    for (uint32_t t = 0; t < div; t++) {
        // do i need like a check if it sles sthan somethi to even run this o rosmethi ?
        DPRINT << "in wirter waitign" << ENDL();
        cb_wait_front(cb_id_dst, 1);
        DPRINT << "in wirter wait finihsed" << ENDL();

        uint32_t l1_read_addr_src = get_read_ptr(cb_id_dst);

        DPRINT << "Tile: " << t << ", Printing the tile values from compute kernel " << ENDL();
        volatile tt_l1_ptr float* ptr = reinterpret_cast<volatile tt_l1_ptr float*>(l1_read_addr_src);
        for (uint32_t j = 0; j < tile_hw; j++) {
            DPRINT << ptr[j] << " ";
        }
        DPRINT << ENDL();

        for (uint32_t i = 0; i < num_rows; i++) {
            uint64_t src_noc_addr = get_noc_addr(current_row + i, dst) + bytes_to_write * t;
            noc_async_write(
                l1_read_addr_src,
                src_noc_addr,
                bytes_to_write);  //  what if bytes here overflows and goes out of bounds ?

            l1_read_addr_src += bytes_to_write;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_dst, 1);
    }
    DPRINT << "Writer Exit" << ENDL();
}
