// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr bool direction = get_compile_time_arg_val(7);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(8);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(9);
constexpr uint32_t ag_worker_cores = get_compile_time_arg_val(10);

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
    uint32_t input_tensor_C = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_C = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 11;

    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    const auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, input_tensor_page_size);

    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    const auto output_tensor_addrgen =
        TensorAccessor(output_tensor_args, output_tensor_address, input_tensor_page_size);

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

    // Create schedule of tiles to be written out by this device

    // Iterate over the schedule of tiles.  There should be a variable which is number of tiles per slice per semaphore
    // In between each semaphore, receive number of tiles per slice per semaphore, and this can allow us to fire off the
    // matmul chunk. However unlike before, it will move on to the next slice after the semaphore to finish the entire
    // pass through K and devices, and then come back to the slice later. We should push out all local data first, get
    // it out on the wire, but in the order that it's supposed to appear.

    // Push out the local slice (make sure it abides by the appropriate schedule)
    // Imagine the input has already been permuted to the appropriate schedule, then the parameters that control
    // this are just input_tile_id_start and end (like before).  It probably needs a number tiles per sync, but that
    // probably goes to the writer, the reader just reads the whole block. Push out our local slice
    uint32_t global_tile_id_start = input_tile_id_start;
    uint32_t global_tile_index = 0;
    uint32_t tiles_per_bh = input_tensor_Wt * input_tensor_Ht;
    bool done = false;

    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        while (!done) {
            uint32_t chunk_tile_index = 0;
            uint32_t next_tile_to_read = global_tile_index;
            // Send local chunk (each chunk is a number of tiles, not contiguous. the stride is the number of ag cores
            // per direction). Across devices, the chunks will add up to a complete k (horizontal) slice of M.  We would
            // expect this to be 1 block_h tall, but it could be arbitrary, and on the matmul side it fires off matmul's
            // until it surpasses block_h.
            while (chunk_tile_index < tiles_per_chunk) {
                cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
                size_t l1_write_addr = get_write_ptr(cb_output_id);
                for (uint32_t j = 0; j < num_tiles_to_write_per_packet; ++j) {
                    uint32_t tile_id = get_next_tile_input(next_tile_to_read, global_tile_id_start, ag_worker_cores);

                    if (tile_id >= tiles_per_bh) {
                        done = true;
                        break;
                    }

                    uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                    l1_write_addr += input_tensor_page_size;
                    next_tile_to_read++;
                    chunk_tile_index++;
                }

                noc_async_read_barrier();
                cb_push_back(cb_output_id, num_tiles_to_write_per_packet);
            }

            // Receive chunks
            uint32_t sem_target = 0;
            while (slices_received < slices_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slices_received, ring_size);

                chunk_tile_index = 0;
                next_tile_to_read = global_tile_index;
                uint32_t slice_tile_end_id =
                    output_tensor_Wt * (output_tensor_Ht - 1) + input_tensor_Wt * (actual_sender_chip_id + 1);

                // Receive the next chunk
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                sem_target++;

                if ((topology == Topology::Linear && writes_expected > 0) ||
                    (topology == Topology::Ring && ((slices_received + 1) < (writes_expected + 1)))) {
                    // read the next chunk out of memory, and put it in CB
                    while (chunk_tile_index < tiles_per_chunk) {
                        cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
                        size_t l1_write_addr = get_write_ptr(cb_output_id);
                        for (uint32_t j = 0; j < num_tiles_to_write_per_packet; ++j) {
                            uint32_t tile_id = get_next_tile_output(
                                next_tile_to_read,
                                global_tile_id_start,
                                ag_worker_cores,
                                input_tensor_Wt,
                                output_tensor_Wt,
                                actual_sender_chip_id);

                            if (tile_id >= slice_tile_end_id) {
                                break;
                            }

                            uint64_t noc_read_addr = get_noc_addr(tile_id, output_tensor_addrgen);
                            noc_async_read(
                                noc_read_addr,
                                l1_write_addr,
                                input_tensor_page_size);  // TODO should change this to output_tensor_page_size

                            l1_write_addr +=
                                input_tensor_page_size;  // TODO should chnage this to output_tensor_page_size
                            next_tile_to_read++;
                            chunk_tile_index++;
                        }

                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, num_tiles_to_write_per_packet);
                    }
                }
                slices_received++;
            }
            global_tile_index = next_tile_to_read;
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        }
        global_tile_id_start += tiles_per_bh;
        global_tile_index = 0;
        done = false;
    }
}
