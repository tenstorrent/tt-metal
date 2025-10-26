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
constexpr uint32_t max_tiles_per_packet = get_compile_time_arg_val(2);
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
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 11;

    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    const auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, input_tensor_page_size);

    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    const auto output_tensor_addrgen =
        TensorAccessor(output_tensor_args, output_tensor_address, input_tensor_page_size);

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
    uint32_t chunks_per_core = div_up(input_tiles_per_core, tiles_per_chunk);

    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        global_tile_index = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_core; chunk_idx++) {
            read_chunk(
                global_tile_index,
                global_tile_id_start,
                cb_output_id,
                input_tiles_per_core,
                tiles_per_chunk,
                max_tiles_per_packet,
                ag_worker_cores,
                input_tensor_addrgen,
                input_tensor_page_size,
                output_tensor_addrgen,
                input_tensor_Wt,
                output_tensor_Wt,
                my_chip_id,
                false);

            // Receive this chunk from all other devices
            uint32_t slices_received = 0;
            uint32_t next_tile_to_read = 0;
            while (slices_received < slices_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slices_received, ring_size);

                // Receive the next chunk
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), slices_received + 1);

                if ((topology == Topology::Linear && writes_expected > 0) ||
                    (topology == Topology::Ring && ((slices_received + 1) < (writes_expected + 1)))) {
                    // read the next chunk out of memory, and put it in CB
                    next_tile_to_read = read_chunk(
                        global_tile_index,
                        global_tile_id_start,
                        cb_output_id,
                        input_tiles_per_core,
                        tiles_per_chunk,
                        max_tiles_per_packet,
                        ag_worker_cores,
                        input_tensor_addrgen,
                        input_tensor_page_size,
                        output_tensor_addrgen,
                        input_tensor_Wt,
                        output_tensor_Wt,
                        actual_sender_chip_id,
                        true);
                }
                slices_received++;
            }
            global_tile_index = next_tile_to_read;
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        }
        global_tile_id_start += tiles_per_bh;
    }
}
