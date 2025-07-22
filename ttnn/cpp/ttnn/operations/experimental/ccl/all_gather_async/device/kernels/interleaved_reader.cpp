// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
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

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    // Push out our local slice
    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_output_id)};

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
                noc_async_read_tile(tile_id, input_tensor_addrgen, l1_write_addr);

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
    constexpr bool output_tensor_is_dram = output_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto output_tensor_addrgen = InterleavedAddrGenFast<output_tensor_is_dram>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_output_id)};
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

    while (slices_received < slices_expected) {
        // Do i expect more from the backward direction?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        // Do i expect more from the forward direction?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), slices_received + 1);
        // Got it
        slices_received++;

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if (direction == 1) {
            sender_chip_id = my_chip_id + slices_received;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - slices_received;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }
        if (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }

        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && writes_expected > 0) ||
            (topology == Topology::Ring && (slices_received < (writes_expected + 1)))) {
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
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

                    cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
                    size_t l1_write_addr = get_write_ptr(cb_output_id);
                    for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                        uint32_t tile_id = output_tile_id_start + row_offset + pages_read_in_row;
                        noc_async_read_tile(tile_id, output_tensor_addrgen, l1_write_addr);

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
        }
    }

    const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem);
    noc_inline_dw_write(dest_noc_addr, 0);
}
