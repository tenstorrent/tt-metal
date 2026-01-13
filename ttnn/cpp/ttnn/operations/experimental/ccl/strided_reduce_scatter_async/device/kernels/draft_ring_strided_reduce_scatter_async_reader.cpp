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
    experimental::ShardedAddrGen<input_tensor_shard_info> input_tensor_addrgen = {
        .bank_base_address = input_tensor_address, .shard_array = input_mapping_table};

    arg_idx += input_rt_increment;
#else
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);
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
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_tensor_addrgen = {
        .bank_base_address = intermediate_tensor_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;
#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);
#endif

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    uint32_t chunk_count = 0;
    uint32_t sem_target = 0;

    uint32_t stride_size = padded_M_tiles / mm_cores_y;

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        if constexpr (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
        uint32_t batch_offset = input_batch_num_pages * b;

        for (uint32_t stride_idx = 0; stride_idx < stride_size; stride_idx++) {
            if constexpr (fuse_op) {
                matmul_receiver.wait_for_matmul_stride(stride_idx);
            }
            // Loop over the slices, starting from the furthest, and working backwards until we get to ourselves
            // Read our local slice at this slice idx into cb_input_id or cb_output_id
            // If we are not the first slice, then read intermediate into the cb_intermediate_id
            // Then reduce those two CB's, and push that to cb_output_id
            // If slices_forwarded in writer is 7, we don't forward anymore and write it to output_buffer
            // Otherwise, the writer will write cb_output_id to the next chip in the forward direction
            for (uint32_t i = 0; i < ring_size; ++i) {
                const bool do_reduce = i != 0;
                uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

                uint32_t actual_slice_idx;
                if (direction) {
                    actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
                } else {
                    actual_slice_idx =
                        slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
                }

                // for now only support single channel
                ASSERT(dim == 3);
                ASSERT(slice_C == 1);

                uint32_t input_chunk_start_tile = get_input_chunk_start_tile(actual_slice_idx, stride_idx);
                uint32_t intermediate_tile_id_start = get_intermediate_chunk_start_tile(actual_slice_idx, stride_idx);
                uint32_t M_blocks_per_core;
                const uint32_t device_block_counts =
                    device_k_block_counts[my_chip_id];  // not sure this can be different for different slices
                const uint32_t device_chunk_widths =
                    device_chunk_widths[my_chip_id];  // not sure this can be different for different slices

                for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
                    for (uint32_t chunk_idx = 0; chunk_idx < device_block_counts; chunk_idx++) {
                        // it is not clear to me that chunks are meaningful here
                        // maybe could simplify to chunk being entire block row for this slice?
                        // on the other hand, this gives some granularity to the matmul -- can fire off faster?
                        uint32_t actual_chunk_w =
                            device_chunk_widths[chunk_idx];  // again, this should probably be the same
                        uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                            input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);
                        uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;

                        read_single_chunk_strided_tiles_from_noc_and_put_into_cb(
                            input_chunk_start_tile,
                            cb_in0,
                            tiles_in_current_chunk,
                            actual_chunk_w,
                            actual_chunk_h,
                            stride_size,
                            tile_granularity,
                            direction);

                        if (do_reduce) {
                            wait_for_semaphore(out_ready_sem, sem_target + 1);
                            read_single_chunk_strided_tiles_from_noc_and_put_into_cb(
                                input_chunk_start_tile,
                                batch_input_tile_offset,
                                cb_intermediate_id,
                                tiles_in_current_chunk,
                                actual_chunk_w,
                                actual_chunk_h,
                                padded_M_tiles / mm_cores_y,
                                tile_granularity,
                                direction);
                            // synchronize after reading chunk into intermediate CB
                            synchronize_workers_and_signal_op(actual_slice_idx);
                        }
                    }
                }

                if (do_reduce && (i == (ring_size - 1))) {
                    // Reset the semaphore before the next batch
                    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
                    sem_target = 0;
                }
            }
        }
    }
}
