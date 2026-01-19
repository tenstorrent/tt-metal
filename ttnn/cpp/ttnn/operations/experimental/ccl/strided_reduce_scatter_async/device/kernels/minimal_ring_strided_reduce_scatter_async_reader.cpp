// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
#include <utility>
#include "api/debug/dprint.h"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(10);
constexpr uint32_t slice_C = get_compile_time_arg_val(11);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(12);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(13);
constexpr uint32_t fuse_op = get_compile_time_arg_val(14);
constexpr uint32_t dim = get_compile_time_arg_val(15);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 16;

#ifdef INPUT_IS_SHARDED
    constexpr uint32_t ct_offset = 7;

    using input_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [input_mapping_table, input_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<input_tensor_shard_info>(get_arg_addr(arg_idx));
    arg_idx += input_rt_increment;
#else
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
#endif

#ifdef INTERMEDIATE_IS_SHARDED
    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    arg_idx += intermediate_rt_increment;
#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
#endif

    // Let's set some particular values for the params used
    const uint32_t M_blocks_per_core = 1;
    const uint32_t chunk_counts_per_width = 1;
    const uint32_t mm_N_blocks_per_slice = 1;
    const uint32_t batch_size = input_tensor_B;
    const uint32_t chunks_per_mm_N_block = 1;
    const int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    uint32_t actual_slice_idx;
    if (direction) {
        actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
    } else {
        actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
    }
    DPRINT << "The reader kernel running its loop." << ENDL();
    DPRINT << "my_chip_id: " << my_chip_id << ENDL();
    DPRINT << "slice_idx: " << slice_idx << ENDL();
    DPRINT << "actual_slice_idx: " << actual_slice_idx << ENDL();
    DPRINT << "slice_Wt: " << slice_Wt << ENDL();
    DPRINT << "slice_Ht: " << slice_Ht << ENDL();
    DPRINT << "slice_C (must be 1): " << slice_C << ENDL();
    DPRINT << "tile_granularity: " << tile_granularity << ENDL();
    DPRINT << "direction: " << (uint32_t)direction << ENDL();
    DPRINT << " chunks_per_sync: " << chunks_per_sync << ENDL();
    DPRINT << " start_tiles_read: " << start_tiles_read << ENDL();
    DPRINT << " start_tiles_to_read: " << start_tiles_to_read << ENDL();
    DPRINT << " start_pages_read_in_row: " << start_pages_read_in_row << ENDL();
    DPRINT << " start_row_offset: " << start_row_offset << ENDL();

    for (uint32_t b = 0; b < batch_size; b++) {
        const uint32_t batch_offset = input_batch_num_pages * b;

        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_block; chunk_idx++) {
                for (uint32_t i = 0; i < ring_size; i++) {
                    const bool do_reduce = i != 0;
                    uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_blocks_per_slice; chunk_piece_idx++) {
                        DPRINT << "batch_size: " << b << " batch_offset:" << batch_offset << " "
                               << " m_block_iter: " << m_block_iter << " i: " << i
                               << " do_reduce: " << (uint32_t)do_reduce << ENDL();
                        DPRINT << "chunk_idx: " << chunk_idx << " chunk_piece_idx: " << chunk_piece_idx << ENDL();
                        DPRINT << "--------------------------------" << ENDL();
                    }
                }
            }
        }
    }
}
