// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr BufferType output_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb_input_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(5);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(6);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(7);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(9);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t ring_size = get_compile_time_arg_val(11);
constexpr uint32_t num_batches = get_compile_time_arg_val(12);
constexpr uint32_t fuse_op = get_compile_time_arg_val(13);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(14);
constexpr bool is_forward = get_compile_time_arg_val(15);
constexpr bool is_first_device_in_direction = get_compile_time_arg_val(16);
constexpr uint32_t num_targets_in_direction = get_compile_time_arg_val(17);
constexpr uint32_t num_intermediate_reduction_steps = get_compile_time_arg_val(18);
constexpr bool do_final_reduction = get_compile_time_arg_val(19);
constexpr uint32_t num_total_reduction_steps = get_compile_time_arg_val(20);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(21);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    constexpr uint32_t ct_idx = 22;

#ifdef INPUT_IS_SHARDED
    constexpr uint32_t ct_offset_one = 7;

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
    constexpr uint32_t ct_offset_one = 0;

    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};
#endif

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset_two = 7;

    constexpr uint32_t inter_start_ct_idx = ct_idx + ct_offset_one;
    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(inter_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(inter_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(inter_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(inter_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(inter_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(inter_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(inter_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_tensor_addrgen = {
        .bank_base_address = intermediate_tensor_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;
#else
    constexpr uint32_t ct_offset_two = 0;

    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};
#endif

#ifdef OUTPUT_IS_SHARDED
    constexpr uint32_t output_start_ct_idx = ct_idx + ct_offset_one + ct_offset_two;
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(output_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(output_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(output_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(output_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                             // padding pages
        get_compile_time_arg_val(output_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                             // calculated
        get_compile_time_arg_val(output_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(output_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_tensor_addrgen = {
        .bank_base_address = output_tensor_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr bool output_tensor_is_dram = output_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto output_tensor_addrgen = InterleavedAddrGenFast<output_tensor_is_dram>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};
#endif

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    constexpr uint32_t slice_Wt = input_tensor_Wt / ring_size;

    constexpr uint32_t batch_num_pages = batch_slice_num_pages * ring_size;

    for (uint32_t b = 0; b < num_batches; b++) {
        if (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        int slice_idx = is_forward ? ring_size - 1 : 0;
        uint32_t batch_offset = batch_num_pages * b;

        // Iterate over the slices in the direction we are going.
        // In forwards direction, count down from slice (ring_size -1) down to (my_chip_id+1), inclusive
        // In backwards cirection, count up from slice 0 to (my_chip_id-1), inclusive
        // After doing all partial reductions and send, there's a final reduction step.
        // If we are not the first device in the direction, do the final reduction.
        // If this device has both FWD and BWD neighbors, the FWD reader will do final reduction first
        // and then signal the BWD reader to do its final reduction.
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            uint32_t input_tile_id_start = slice_idx * slice_Wt + batch_offset;
            uint32_t intermediate_tile_id_start = slice_idx * slice_Wt;
            uint32_t stride_Wt = input_tensor_Wt;
            uint32_t pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t intermediate_pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t intermediate_row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;

            if constexpr (is_first_device_in_direction) {
                // We have no incoming slices, so forward directly to writer.
                uint32_t cb_in0 = cb_reader_output_id;
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                    cb_reserve_back(cb_in0, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);
                        l1_write_addr += input_tensor_page_size;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }
                    tiles_read += num_pages_to_read;
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                }
            } else {
                // I have incoming slices, so write my output to compute kernel and read intermediate input
                uint32_t cb_in0 = cb_input_id;
                // Wait on output semaphore
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), iter + 1);

                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                    cb_reserve_back(cb_in0, tile_granularity);

                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);
                        l1_write_addr += input_tensor_page_size;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    tiles_read += num_pages_to_read;

                    // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, tile_granularity);

                    l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        uint32_t tile_id =
                            intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, intermediate_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);
                        l1_write_addr += input_tensor_page_size;
                        intermediate_pages_read_in_row++;
                        if (intermediate_pages_read_in_row >= slice_Wt) {
                            intermediate_row_offset += stride_Wt;
                            intermediate_pages_read_in_row = 0;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                    cb_push_back(cb_intermediate_id, tile_granularity);
                }
            }

            // Next slice idx
            if constexpr (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do the final reduction. Synchronize with other direction.
        if constexpr (do_final_reduction) {
            bool accumulate_output =
                false;  // If true, output += intermediate. Otherwise, output = input + intermediate
            constexpr bool use_output_tensor_addrgen = sync_with_other_direction && !is_forward;
            if constexpr (use_output_tensor_addrgen) {
                /**
                 * If two cores are doing final reduction, BWD core will accumulate output with
                 * incoming BWD intermediate. Use slice_idx=0 to index into output buffer, and
                 * use output address generator.
                 */
                accumulate_output = true;
                // Wait for FWD writer to signal that it has done its final reduction
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), 1);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), 0);
            }

            /**
             * For final reduction middle chips, we are receiving two inputs at the same time.
             * Our neighbor behind us (my_chip_id-1 writing FWD) will always write to (my_chip_id) slice.
             * Our neighbor in front of us (my_chip_id+1 writing BWD) is signaled by the FWD writer
             * that a slot has freed up. If my first write was forward, (my_chip_id+1) will write to
             * slot (ring_size-1). If my first write was backward, (my_chip_id-1) will write to slot 0.
             *
             * My first write was FWD if I'm on the left half of the ring, and BWD if I'm on the right half.
             */
            uint32_t slice_idx = my_chip_id;
            uint32_t intermediate_slice_idx = my_chip_id;

            if constexpr (!is_forward) {
                constexpr bool my_first_write_was_fwd = my_chip_id < ring_size / 2;
                if constexpr (my_first_write_was_fwd) {
                    intermediate_slice_idx = ring_size - 1;
                } else {
                    intermediate_slice_idx = 0;
                }
            }

            uint32_t input_tile_id_start = slice_idx * slice_Wt + batch_offset;
            uint32_t intermediate_tile_id_start = intermediate_slice_idx * slice_Wt;
            uint32_t stride_Wt = input_tensor_Wt;
            uint32_t intermediate_stride_Wt = input_tensor_Wt;
            uint32_t pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t intermediate_pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t intermediate_row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;
            uint32_t cb_in0 = cb_input_id;

            if (accumulate_output) {
                input_tile_id_start = b * batch_slice_num_pages;  // output batch offset
                pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
                row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * slice_Wt;
                stride_Wt = slice_Wt;
            }
            // Wait on output semaphore
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), num_targets_in_direction + 1);
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                cb_reserve_back(cb_in0, tile_granularity);

                uint32_t l1_write_addr = get_write_ptr(cb_in0);
                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                    uint64_t noc_read_addr = use_output_tensor_addrgen ? get_noc_addr(tile_id, output_tensor_addrgen)
                                                                       : get_noc_addr(tile_id, input_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);
                    l1_write_addr += input_tensor_page_size;
                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }
                }
                tiles_read += num_pages_to_read;

                // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                cb_reserve_back(cb_intermediate_id, tile_granularity);

                l1_write_addr = get_write_ptr(cb_intermediate_id);
                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    uint32_t tile_id =
                        intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                    uint64_t noc_read_addr = get_noc_addr(tile_id, intermediate_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);
                    l1_write_addr += input_tensor_page_size;
                    intermediate_pages_read_in_row++;
                    if (intermediate_pages_read_in_row >= slice_Wt) {
                        intermediate_row_offset += intermediate_stride_Wt;
                        intermediate_pages_read_in_row = 0;
                    }
                }

                noc_async_read_barrier();
                cb_push_back(cb_in0, tile_granularity);
                cb_push_back(cb_intermediate_id, tile_granularity);
            }
        }

        // Reset my output ready semaphore
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
    }
}
