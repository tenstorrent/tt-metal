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

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr bool direction = get_compile_time_arg_val(7);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(8);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(9);
constexpr uint32_t ag_worker_cores = get_compile_time_arg_val(10);

uint32_t get_next_tile_to_read(
    uint32_t local_tile_index, uint32_t input_start_tile_index, uint32_t ag_parallel_factor) {
    // Imagine the input is already permuted (this has nothing to do with our all gather, it's just ordering the output
    // of all gather such it will be ideal for matmul) We split up the work evenly amongst the all gather cores.
    // Probably the best way is just to round robin through the input amongst the various all gather cores.  Ignore
    // direction since you send the same thing forward and backward.  For now just send the whole thing in that order,
    // we can add finer grain fidelity to correspond to the syncs for matmul.  Right now just sync once when we reach
    // the end of the buffer.
    return input_start_tile_index + local_tile_index * ag_parallel_factor;
}

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
    uint32_t current_tile_index = 0;
    uint32_t tiles_per_bh = input_tensor_Wt * input_tensor_Ht;
    bool done = false;

    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        while (!done) {
            cb_reserve_back(cb_output_id, num_tiles_to_write_per_packet);
            size_t l1_write_addr = get_write_ptr(cb_output_id);
            for (uint32_t j = 0; j < num_tiles_to_write_per_packet; ++j) {
                uint32_t tile_id = get_next_tile_to_read(current_tile_index, global_tile_id_start, ag_worker_cores);

                if (tile_id >= tiles_per_bh) {
                    done = true;
                    break;
                }

                uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                l1_write_addr += input_tensor_page_size;
                current_tile_index++;
            }

            noc_async_read_barrier();
            cb_push_back(cb_output_id, num_tiles_to_write_per_packet);
        }
        global_tile_id_start += tiles_per_bh;
        current_tile_index = 0;
        done = false;
    }
}
