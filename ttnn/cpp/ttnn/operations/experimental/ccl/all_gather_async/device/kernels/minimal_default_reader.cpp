// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType output_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(4);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(8));
constexpr bool direction = get_compile_time_arg_val(9);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(10);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(11);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 12;

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
    constexpr uint32_t ct_offset = 0;

    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_addrgen = {
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_output_id)};
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
    constexpr bool output_tensor_is_dram = output_buffer_type == tt::tt_metal::BufferType::DRAM;
    const InterleavedAddrGenFast<output_tensor_is_dram> output_tensor_addrgen = {
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_output_id)};
#endif

    OpSignaler op_signaler;
    uint32_t self_write_done_semaphore_addr;
    if constexpr (fuse_op) {
        self_write_done_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        op_signaler = OpSignaler(arg_idx);
    }

    // Push out our local slice
    uint32_t tiles_read = input_tile_id_start;
    uint32_t tiles_to_read = input_tile_id_end;
    uint32_t output_tile_id_start = 0;
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        while (tiles_read < tiles_to_read) {
            uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
            uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
            size_t l1_write_addr = get_write_ptr(cb_output_id);
            for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                uint32_t tile_id = output_tile_id_start + tiles_read;
                uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                l1_write_addr += input_tensor_page_size;
                tiles_read++;
            }

            noc_async_read_barrier();
            cb_push_back(cb_output_id, num_tiles_to_write_per_packet);
        }
        tiles_read = input_tile_id_start;
        tiles_to_read = input_tile_id_end;
        output_tile_id_start += input_tensor_Wt * input_tensor_Ht;
    }

    uint32_t slices_received = 0;
    uint32_t slices_expected = 0;
    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (direction == 1) {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
        } else {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_backward_direction - 1;
        } else {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    uint32_t chunk_count = 0;
    uint32_t sem_target = 0;
    while (slices_received < slices_expected) {
        // Do i expect more from the backward direction?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        // Do i expect more from the forward direction?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if (direction == 1) {
            sender_chip_id = my_chip_id + slices_received + 1;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - (slices_received + 1);
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && writes_expected > 0) ||
            (topology == Topology::Ring && ((slices_received + 1) < (writes_expected + 1)))) {
            // read the next backward slice out of memory, and put it in CB
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;

            uint32_t output_tile_id_start = 0;
            uint32_t pages_read_in_row = start_pages_read_in_row;
            uint32_t row_offset = start_row_offset;
            uint32_t slice_Wt = input_tensor_Wt;
            uint32_t stride_Wt = output_tensor_Wt;
            if (gather_dim == 3) {
                output_tile_id_start = actual_sender_chip_id * input_tensor_Wt;
            } else {
                output_tile_id_start = actual_sender_chip_id * input_tensor_Ht * input_tensor_Wt;
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                chunk_count = 0;
                while (tiles_read < tiles_to_read) {
                    if (chunk_count % chunks_per_sync == 0) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                        sem_target++;
                    }
                    chunk_count++;

                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

                    cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
                    size_t l1_write_addr = get_write_ptr(cb_output_id);
                    for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                        uint32_t tile_id = output_tile_id_start + row_offset + pages_read_in_row;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, output_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                        l1_write_addr += input_tensor_page_size;
                        tiles_read++;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, num_tiles_to_write_per_packet);
                }
                pages_read_in_row = start_pages_read_in_row;
                row_offset = start_row_offset;
                tiles_read = input_tile_id_start;
                tiles_to_read = input_tile_id_end;
                output_tile_id_start += output_tensor_Wt * output_tensor_Ht;
            }
        } else {
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                chunk_count = 0;
                tiles_read = input_tile_id_start;
                tiles_to_read = input_tile_id_end;
                while (tiles_read < tiles_to_read) {
                    if (chunk_count % chunks_per_sync == 0) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                        sem_target++;
                    }
                    chunk_count++;
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                    tiles_read += num_tiles_to_read;
                }
                tiles_read = input_tile_id_start;
                tiles_to_read = input_tile_id_end;
            }
        }

        slices_received++;
        if (fuse_op) {
            // Signal matmul to go
            if (direction == 1 && slices_received == 1) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(self_write_done_semaphore_addr), 1);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(self_write_done_semaphore_addr), 0);
            }
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
    }

    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
