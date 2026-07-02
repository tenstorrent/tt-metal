// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t src0_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_per_input_block = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_blocks_across_width = get_compile_time_arg_val(4);
    constexpr uint32_t output_element_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_cols_per_input_block = get_compile_time_arg_val(6);
    constexpr uint32_t num_cols_per_output_block = get_compile_time_arg_val(7);
    constexpr uint32_t num_shards = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(11);
    constexpr uint32_t tile_width = get_compile_time_arg_val(12);
    constexpr uint32_t output_tensor_width = get_compile_time_arg_val(13);
    constexpr uint32_t output_tensor_height = get_compile_time_arg_val(14);

    constexpr auto dst_args = TensorAccessorArgs<15>();
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);
    constexpr auto src0_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const auto accessor_src = TensorAccessor(src0_args, src0_addr);

    Noc noc;
    DataflowBuffer cb_out(cb_id_out0);

    auto write_tiles_in_current_block = [&](uint32_t block_height_index,
                                            uint32_t width_wise_output_block_start_index,
                                            uint32_t num_unpadded_cols_per_input_block,
                                            uint32_t num_cols_already_processed_in_first_output_block,
                                            uint32_t num_rows_to_write) {
        cb_out.wait_front(num_tiles_per_input_block);

        uint32_t base_l1_read_addr = cb_out.get_read_ptr();

        for (uint32_t j = 0; j < num_rows_to_write; ++j) {
            uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;

            uint32_t num_rows_already_processed = block_height_index * tile_height + j;
            uint32_t num_pages_already_processed_in_previous_rows =
                num_rows_already_processed * num_output_blocks_across_width;
            uint32_t output_page_id =
                num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;

            uint32_t num_cols_remaining_in_current_output_block =
                num_cols_per_output_block - num_cols_already_processed_in_first_output_block;
            uint32_t output_offset_within_page_in_bytes =
                num_cols_already_processed_in_first_output_block * output_element_size;

            uint32_t num_input_cols_processed = 0;
            while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
                uint32_t num_cols_to_write = std::min(
                    num_unpadded_cols_per_input_block - num_input_cols_processed,
                    num_cols_remaining_in_current_output_block);
                uint32_t num_bytes_to_write = num_cols_to_write * output_element_size;

                CoreLocalMem<uint32_t> src(current_l1_read_addr);
                noc.async_write(
                    src,
                    accessor_dst,
                    num_bytes_to_write,
                    {.offset_bytes = 0},
                    {.page_id = output_page_id, .offset_bytes = output_offset_within_page_in_bytes});

                num_input_cols_processed += num_cols_to_write;
                current_l1_read_addr += num_bytes_to_write;
                output_page_id++;
                num_cols_remaining_in_current_output_block = num_cols_per_output_block;
                output_offset_within_page_in_bytes = 0;
            }
        }

        noc.async_write_barrier();
        cb_out.pop_front(num_tiles_per_input_block);
    };

    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_src.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end();
             page_iter += num_tiles_per_input_block) {
            uint32_t page_id = page_iter->page_id();
            uint32_t height_wise_input_block_index = page_id / num_tiles_per_row;
            uint32_t tile_index_width = page_id % num_tiles_per_row;
            uint32_t width_wise_input_block_index = tile_index_width / num_tiles_per_input_block;

            uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
            uint32_t last_column_tensor_index_in_block =
                (tile_index_width + num_tiles_per_input_block) * tile_width - 1;
            if (last_column_tensor_index_in_block >= output_tensor_width) {
                num_unpadded_cols_per_input_block = (output_tensor_width - tile_index_width * tile_width);
            }
            uint32_t last_row_tensor_index_in_block = (height_wise_input_block_index + 1) * tile_height - 1;
            uint32_t num_rows_to_write = tile_height;
            if (last_row_tensor_index_in_block >= output_tensor_height) {
                num_rows_to_write = (output_tensor_height - height_wise_input_block_index * tile_height);
            }
            uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
            uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
            uint32_t num_cols_already_processed_in_first_output_block =
                input_block_global_col_index % num_cols_per_output_block;
            write_tiles_in_current_block(
                height_wise_input_block_index,
                width_wise_output_block_start_index,
                num_unpadded_cols_per_input_block,
                num_cols_already_processed_in_first_output_block,
                num_rows_to_write);
        }
    }
}
