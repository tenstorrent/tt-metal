// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id_dir0 = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);  // 1 is forward, 0 is backward
constexpr uint32_t gather_dim = get_compile_time_arg_val(9);
constexpr uint32_t input_batch_head_count = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_Ht = get_compile_time_arg_val(12);
constexpr uint32_t input_tensor_C = get_compile_time_arg_val(13);
constexpr uint32_t output_tensor_Wt = get_compile_time_arg_val(14);
constexpr uint32_t output_tensor_Ht = get_compile_time_arg_val(15);
constexpr uint32_t output_tensor_C = get_compile_time_arg_val(16);
constexpr uint32_t input_tile_id_start = get_compile_time_arg_val(17);
constexpr uint32_t input_tile_id_end = get_compile_time_arg_val(18);
constexpr uint32_t start_pages_read_in_row = get_compile_time_arg_val(19);
constexpr uint32_t start_row_offset = get_compile_time_arg_val(20);
constexpr bool fuse_op = get_compile_time_arg_val(21);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(22);
constexpr uint32_t reverse = get_compile_time_arg_val(23) == 1;

constexpr uint32_t cb_output_id_dir1 = get_compile_time_arg_val(24);

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem_dir0 = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_dir1 = get_arg_val<uint32_t>(arg_idx++);
    // DPRINT << "Reader arg_idx" << (uint32_t)arg_idx << ENDL();
    constexpr uint32_t ct_idx = 25;

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
    const auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);
#endif

#ifdef OUTPUT_IS_SHARDED
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_tensor_addrgen = {
        .bank_base_address = output_tensor_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    const auto output_tensor_addrgen = TensorAccessor(output_tensor_args, output_tensor_address, page_size);
#endif

    OpSignaler op_signaler;
    uint32_t self_write_done_semaphore_addr;
    if constexpr (fuse_op) {
        self_write_done_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        op_signaler = OpSignaler(arg_idx);
    }

    // Push out our local slice
    uint32_t tiles_read_dir0 = input_tile_id_start;
    uint32_t tiles_read_dir1 = input_tile_id_start;
    uint32_t tiles_to_read_dir0 = input_tile_id_end;
    uint32_t tiles_to_read_dir1 = input_tile_id_end;
    uint32_t output_tile_id_start_dir0 = 0;
    uint32_t output_tile_id_start_dir1 = 0;
    // DPRINT << "Reader Input Start " << (uint32_t)input_tile_id_start << " ~ " << (uint32_t)input_tile_id_end << ENDL();
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        while (tiles_read_dir0 < tiles_to_read_dir0) {
            uint32_t tiles_remaining_to_read = tiles_to_read_dir0 - tiles_read_dir0;
            uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            // DPRINT << "Reader cb_reserve_back cb_output_id_dir0 : " << (uint32_t)cb_output_id_dir0 << ENDL();
            // DPRINT << "Reader cb_reserve_back num_tiles_to_read : " << (uint32_t)num_tiles_to_read << ENDL();
            cb_reserve_back(cb_output_id_dir0, num_tiles_to_write_per_packet);
            size_t l1_write_addr = get_write_ptr(cb_output_id_dir0);
            for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                uint32_t tile_id = output_tile_id_start_dir0 + tiles_read_dir0;
                uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                // DPRINT << "Reader noc_async_read" << ENDL();
                noc_async_read(noc_read_addr, l1_write_addr, page_size);

                l1_write_addr += page_size;
                tiles_read_dir0++;
            }
            // DPRINT << "Reader noc_async_read_barrier" << ENDL();
            noc_async_read_barrier();
            // DPRINT << "Reader cb_push_back" << ENDL();
            cb_push_back(cb_output_id_dir0, num_tiles_to_write_per_packet);
        }
        tiles_read_dir0 = input_tile_id_start;
        tiles_to_read_dir0 = input_tile_id_end;
        output_tile_id_start_dir0 += input_tensor_Wt * input_tensor_Ht;
    }
    // DPRINT << "Reader Input Done" << ENDL();

    uint32_t slices_received_dir0 = 0;
    uint32_t slices_expected_dir0 = 0;
    uint32_t writes_expected_dir0 = 0;
    uint32_t slices_received_dir1 = 0;
    uint32_t slices_expected_dir1 = 0;
    uint32_t writes_expected_dir1 = 0;
    if constexpr (topology == Topology::Linear) {
        // if constexpr (direction == 1) {
            slices_expected_dir1 = num_targets_forward_direction;
            writes_expected_dir1 = num_targets_backward_direction ? num_targets_forward_direction : 0;
        // } else {
            slices_expected_dir0 = num_targets_backward_direction;
            writes_expected_dir0 = num_targets_forward_direction ? num_targets_backward_direction : 0;
        // }
    } else if constexpr (topology == Topology::Ring) {
        // if constexpr (direction == 1) {
            slices_expected_dir1 = num_targets_backward_direction;
            writes_expected_dir1 = num_targets_backward_direction - 1;
        // } else {
            slices_expected_dir0 = num_targets_forward_direction;
            writes_expected_dir0 = num_targets_forward_direction - 1;
        // }
    }

    uint32_t chunk_count_dir0 = 0;
    uint32_t chunk_count_dir1 = 0;
    uint32_t sem_target_dir0 = 0;
    uint32_t sem_target_dir1 = 0;

    // DPRINT << "Reader my_chip_id : " << (uint32_t)my_chip_id << ENDL();
    while ((slices_received_dir1 < slices_expected_dir1) || (slices_received_dir0 < slices_expected_dir0)) {
        // Do i expect more from the backward direction?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        // Do i expect more from the forward direction?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)
        // DPRINT << "Reader slices_received_dir0 < slices_expected_dir0 " << slices_received_dir0 << " < " << slices_expected_dir0 << ENDL();
        // DPRINT << "Reader slices_received_dir1 < slices_expected_dir1 " << slices_received_dir1 << " < " << slices_expected_dir1 << ENDL();
        int sender_chip_id_dir0;
        int sender_chip_id_dir1;
        uint32_t actual_sender_chip_id_dir0;
        uint32_t actual_sender_chip_id_dir1;

        if (slices_received_dir1 < slices_expected_dir1) {
            sender_chip_id_dir1 = my_chip_id + slices_received_dir1 + 1;
            actual_sender_chip_id_dir1 = (sender_chip_id_dir1 >= (int)ring_size) ? sender_chip_id_dir1 - ring_size : sender_chip_id_dir1;
            if constexpr (reverse) {
                actual_sender_chip_id_dir1 = (ring_size - 1) - actual_sender_chip_id_dir1;
            }
            // DPRINT << " - Reader actual_sender_chip_id_dir1 : " << (uint32_t)actual_sender_chip_id_dir1 << ENDL();
        }

        if (slices_received_dir0 < slices_expected_dir0) {
            sender_chip_id_dir0 = my_chip_id - (slices_received_dir0 + 1);
            actual_sender_chip_id_dir0 = (sender_chip_id_dir0 < 0) ? ring_size + sender_chip_id_dir0 : sender_chip_id_dir0;
            if constexpr (reverse) {
                actual_sender_chip_id_dir0 = (ring_size - 1) - actual_sender_chip_id_dir0;
            }
            // DPRINT << " - Reader actual_sender_chip_id_dir0 : " << (uint32_t)actual_sender_chip_id_dir0 << ENDL();
        }

        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && (writes_expected_dir0 > 0 || writes_expected_dir1 > 0)) ||
            (topology == Topology::Ring   && ((slices_received_dir0 + 1) < (writes_expected_dir0 + 1) ||
                                            (slices_received_dir1 + 1) < (writes_expected_dir1 + 1)))) {
            // read the next backward slice out of memory, and put it in CB
            if (slices_received_dir0 + 1 < writes_expected_dir0 + 1) {
                tiles_read_dir0 = input_tile_id_start;
                tiles_to_read_dir0 = input_tile_id_end;
            } else {
                tiles_read_dir0 = 0;
                tiles_to_read_dir0 = 0;
            }

            if (slices_received_dir1 + 1 < writes_expected_dir1 + 1) {
                tiles_read_dir1 = input_tile_id_start;
                tiles_to_read_dir1 = input_tile_id_end;
            } else {
                tiles_read_dir1 = 0;
                tiles_to_read_dir1 = 0;
            }

            // DPRINT << " - Reader tiles_read_dir0 : " << (uint32_t)tiles_read_dir0 << " ~ " << (uint32_t)tiles_to_read_dir0 << ENDL();
            // DPRINT << " - Reader tiles_read_dir1 : " << (uint32_t)tiles_read_dir1 << " ~ " << (uint32_t)tiles_to_read_dir1 << ENDL();

            uint32_t output_tile_id_start_dir0 = 0;
            uint32_t output_tile_id_start_dir1 = 0;
            uint32_t pages_read_in_row_dir0 = start_pages_read_in_row;
            uint32_t pages_read_in_row_dir1 = start_pages_read_in_row;
            uint32_t row_offset_dir0 = start_row_offset;
            uint32_t row_offset_dir1 = start_row_offset;
            uint32_t slice_Wt = input_tensor_Wt;
            uint32_t stride_Wt = output_tensor_Wt;
            if constexpr (gather_dim == 3) {
                output_tile_id_start_dir0 = actual_sender_chip_id_dir0 * input_tensor_Wt;
                output_tile_id_start_dir1 = actual_sender_chip_id_dir1 * input_tensor_Wt;
            } else if constexpr (gather_dim == 2) {
                output_tile_id_start_dir0 = actual_sender_chip_id_dir0 * input_tensor_Ht * input_tensor_Wt;
                output_tile_id_start_dir1 = actual_sender_chip_id_dir1 * input_tensor_Ht * input_tensor_Wt;
            } else if constexpr (gather_dim == 1) {
                output_tile_id_start_dir0 = actual_sender_chip_id_dir0 * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
                output_tile_id_start_dir1 = actual_sender_chip_id_dir1 * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
            } else {
                output_tile_id_start_dir0 =
                    actual_sender_chip_id_dir0 * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
                output_tile_id_start_dir1 =
                    actual_sender_chip_id_dir1 * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
            }

            uint32_t num_channels_processed_in_current_batch = 0;
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                chunk_count_dir0 = 0;
                chunk_count_dir1 = 0;

                // DPRINT << " - Reader Processing batch head " << (uint32_t)bh_idx << ENDL();
                while ((tiles_read_dir0 < tiles_to_read_dir0) || (tiles_read_dir1 < tiles_to_read_dir1)) {
                    // DPRINT << "  - Reader tiles_read_dir0 < tiles_to_read_dir0 " << (uint32_t)tiles_read_dir0 << " < " << (uint32_t)tiles_to_read_dir0 << ENDL();
                    // DPRINT << "  - Reader tiles_read_dir1 < tiles_to_read_dir1 " << (uint32_t)tiles_read_dir1 << " < " << (uint32_t)tiles_to_read_dir1 << ENDL();
                    if (tiles_read_dir1 < tiles_to_read_dir1) {
                        if (chunk_count_dir1 % chunks_per_sync == 0) {
                            // forward path semaphore
                            // DPRINT << "   - Reader dir1 sem_wait : " << sem_target_dir1 + 1 << ENDL();
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir1), sem_target_dir1 + 1);
                            // DPRINT << "   - Reader dir1 sem_wait End : " << sem_target_dir1 + 1 << ENDL();
                            sem_target_dir1++;
                        }
                        chunk_count_dir1++;

                        uint32_t tiles_remaining_to_read = tiles_to_read_dir1 - tiles_read_dir1;
                        uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                        // DPRINT << "   - Reader cb_reserve_back dir1" << ENDL();
                        cb_reserve_back(cb_output_id_dir1, num_tiles_to_write_per_packet);
                        size_t l1_write_addr = get_write_ptr(cb_output_id_dir1);
                        for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                            uint32_t tile_id = output_tile_id_start_dir1 + row_offset_dir1 + pages_read_in_row_dir1;
                            uint64_t noc_read_addr = get_noc_addr(tile_id, output_tensor_addrgen);
                            noc_async_read(noc_read_addr, l1_write_addr, page_size);

                            l1_write_addr += page_size;
                            tiles_read_dir1++;

                            pages_read_in_row_dir1++;
                            if (pages_read_in_row_dir1 >= slice_Wt) {
                                row_offset_dir1 += stride_Wt;
                                pages_read_in_row_dir1 = 0;
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id_dir1, num_tiles_to_write_per_packet);
                        // DPRINT << "   - Reader cb_push_back dir1" << ENDL();
                    }
                    if (tiles_read_dir0 < tiles_to_read_dir0) {
                        if (chunk_count_dir0 % chunks_per_sync == 0) {
                            // backward path semaphore
                            // DPRINT << "   - Reader dir0 sem_wait : " << sem_target_dir0 + 1 << ENDL();
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir0), sem_target_dir0 + 1);
                            // DPRINT << "   - Reader dir0 sem_wait End : " << sem_target_dir0 + 1 << ENDL();
                            sem_target_dir0++;
                        }
                        chunk_count_dir0++;

                        uint32_t tiles_remaining_to_read = tiles_to_read_dir0 - tiles_read_dir0;
                        uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                        
                        // DPRINT << "   - Reader cb_reserve_back dir0" << ENDL();
                        cb_reserve_back(cb_output_id_dir0, num_tiles_to_write_per_packet);
                        size_t l1_write_addr = get_write_ptr(cb_output_id_dir0);
                        for (uint32_t j = 0; j < num_tiles_to_read; j++) {
                            uint32_t tile_id = output_tile_id_start_dir0 + row_offset_dir0 + pages_read_in_row_dir0;
                            uint64_t noc_read_addr = get_noc_addr(tile_id, output_tensor_addrgen);
                            noc_async_read(noc_read_addr, l1_write_addr, page_size);
                            l1_write_addr += page_size;
                            tiles_read_dir0++;

                            pages_read_in_row_dir0++;
                            if (pages_read_in_row_dir0 >= slice_Wt) {
                                row_offset_dir0 += stride_Wt;
                                pages_read_in_row_dir0 = 0;
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id_dir0, num_tiles_to_write_per_packet);
                        // DPRINT << "   - Reader cb_push_back dir0" << ENDL();
                    }
                    // noc_async_read_barrier();
                    // if (tiles_read_dir0 < tiles_to_read_dir0) {
                    //     cb_push_back(cb_output_id_dir0, num_tiles_to_write_per_packet);
                    // }
                    // if (tiles_read_dir1 < tiles_to_read_dir1) {
                    //     cb_push_back(cb_output_id_dir1, num_tiles_to_write_per_packet);
                    // }
                }
                // DPRINT << " - Reader Finished batch head " << (uint32_t)bh_idx << ENDL();
                num_channels_processed_in_current_batch++;
                if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                    output_tile_id_start_dir0 +=
                        output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
                    output_tile_id_start_dir1 +=
                        output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
                } else {
                    output_tile_id_start_dir0 += output_tensor_Wt * output_tensor_Ht;
                    output_tile_id_start_dir1 +=  output_tensor_Wt * output_tensor_Ht;
                }

                if (num_channels_processed_in_current_batch == input_tensor_C) {
                    num_channels_processed_in_current_batch = 0;
                }

                pages_read_in_row_dir0 = start_pages_read_in_row;
                pages_read_in_row_dir1 = start_pages_read_in_row;
                row_offset_dir0 = start_row_offset;
                row_offset_dir1 = start_row_offset;
                tiles_read_dir0 = input_tile_id_start;
                tiles_read_dir1 = input_tile_id_start;
            }
        } else {
            // DPRINT << " - Reader Skipping forward step, no more writes expected" << ENDL();
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                chunk_count_dir0 = 0;
                chunk_count_dir1 = 0;

                if (!(slices_received_dir0 + 1 < writes_expected_dir0 + 1)) {
                    tiles_read_dir0 = input_tile_id_start;
                    tiles_to_read_dir0 = input_tile_id_end;
                } else {
                    tiles_read_dir0 = 0;
                    tiles_to_read_dir0 = 0;
                }
                if (!(slices_received_dir1 + 1 < writes_expected_dir1 + 1)) {
                    tiles_read_dir1 = input_tile_id_start;
                    tiles_to_read_dir1 = input_tile_id_end;
                } else {
                    tiles_read_dir1 = 0;
                    tiles_to_read_dir1 = 0;
                }

                while ((tiles_read_dir0 < tiles_to_read_dir0) || (tiles_read_dir1 < tiles_to_read_dir1)) {
                    if (tiles_read_dir0 < tiles_to_read_dir0) {
                        if (chunk_count_dir0 % chunks_per_sync == 0) {
                            // DPRINT << "   - Reader dir0 sem_wait (skip) : " << sem_target_dir0 + 1 << ENDL();
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir0), sem_target_dir0 + 1);
                            // DPRINT << "   - Reader dir0 sem_wait (skip) : " << sem_target_dir0 + 1 << " done" << ENDL();
                            sem_target_dir0++;
                        }
                        chunk_count_dir0++;
                        uint32_t tiles_remaining_to_read = tiles_to_read_dir0 - tiles_read_dir0;
                        uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                        tiles_read_dir0 += num_tiles_to_read;
                    }
                    if (tiles_read_dir1 < tiles_to_read_dir1) {
                        if (chunk_count_dir1 % chunks_per_sync == 0) {
                            // DPRINT << "   - Reader dir1 sem_wait (skip) : " << sem_target_dir1 + 1 << ENDL();
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir1), sem_target_dir1 + 1);
                            // DPRINT << "   - Reader dir1 sem_wait (skip) : " << sem_target_dir1 + 1 << " done" << ENDL();
                            sem_target_dir1++;
                        }
                        chunk_count_dir1++;
                        uint32_t tiles_remaining_to_read = tiles_to_read_dir1 - tiles_read_dir1;
                        uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                        tiles_read_dir1 += num_tiles_to_read;
                    }
                }
                if (!(slices_received_dir0 + 1 < writes_expected_dir0 + 1)) {
                    tiles_read_dir0 = input_tile_id_start;
                    tiles_to_read_dir0 = input_tile_id_end;
                } else {
                    tiles_read_dir0 = 0;
                    tiles_to_read_dir0 = 0;
                }

                if (!(slices_received_dir1 + 1 < writes_expected_dir1 + 1)) {
                    tiles_read_dir1 = input_tile_id_start;
                    tiles_to_read_dir1 = input_tile_id_end;
                } else {
                    tiles_read_dir1 = 0;
                    tiles_to_read_dir1 = 0;
                }
            }
        }

        if (slices_received_dir0 < slices_expected_dir0) {
            slices_received_dir0++;
        }
        
        if (slices_received_dir1 < slices_expected_dir1) {
            slices_received_dir1++;
        }

        if constexpr (fuse_op) {
            // Signal matmul to go
            // if (direction == 1 && slices_received == 1) {
            //     noc_semaphore_wait_min(
            //         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(self_write_done_semaphore_addr), 1);
            //     noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(self_write_done_semaphore_addr), 0);
            // }
            // op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
    }
    
    // DPRINT << "Reader Done" << ENDL();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir0), 0);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_dir1), 0);
}
