// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

#include "fetch_gate_up.h"

// Input-broadcaster kernel (runs on the second core, {1,0}).
//
// Uses the *other* NoC (NoC 1) so it runs concurrently with the expert-id sender
// on {0,0} (NoC 0).
//
//   1. Reads all Kt tiles of input_tensor into cb_input.
//   2. Multicasts the tiles to every other core's L1 (same cb_input address).
//   3. Sets + multicasts the input-ready semaphore to signal the other cores.
//   4. Publishes cb_input to this core's compute kernel.
//   5. Runs the per-expert reader loop as a (non-leader) receiver: fetches this core's
//      gate_up + down slices and receives the broadcast activation for the down matmul.
//
// NoC 1 multicasts traverse from high to low coordinates, so the host passes the
// multicast rectangle with start = bottom-right corner, end = top-left corner.
//
// Compile-time args:
//   0: cb_input         (activation tiles; broadcast to all cores)
//   1: input_page_size  (bytes per tile of input_tensor)
//   2: input_num_pages  (Kt == H / 32)
//   3: sem_input_id     (input-ready semaphore)
//   4: sem_id           (expert-ids-ready / sequencing semaphore)
//   5: num_active       (routing-selected experts to run)
//   6: cb_weights       (this core's per-expert gate_up slice)
//   7: k_tiles          (H / 32)
//   8: i_tiles          (I / 32)
//   9: gate_up_tile_bytes
//   10: cb_bcast        (broadcast hit-expert ids, read by the weight fetch)
//   11: cb_down_w       (this core's per-expert down slice)
//   12: cb_act          (gathered activation)
//   13: down_slice_tiles
//   14: down_tile_bytes
//   15: act_tile_bytes
//   16: num_producers
//   17: sem_gather
//   18: sem_bcast
//   19: num_weights
//   20: cb_rscalar
//   21+: TensorAccessorArgs(input_tensor), TensorAccessorArgs(gate_up), TensorAccessorArgs(down)
//   then: gate_up base addresses (one per expert), then down base addresses (one per expert)
//
// Runtime args:
//   0: input_tensor base address
//   1: mcast_start_x   2: mcast_start_y
//   3: mcast_end_x     4: mcast_end_y
//   5: num_dests       (number of receiver cores = total cores - 1)
//   6: col_start_tile  (this core's first output tile)
void kernel_main() {
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t input_num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(3);
    constexpr uint32_t sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t num_active = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t gate_up_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(10);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(11);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(12);
    constexpr uint32_t down_slice_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t down_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_producers = get_compile_time_arg_val(16);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(17);
    constexpr uint32_t sem_bcast_id = get_compile_time_arg_val(18);
    constexpr uint32_t num_weights = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rscalar_id = get_compile_time_arg_val(20);

    constexpr auto input_args = TensorAccessorArgs<21>();
    constexpr auto gate_up_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto down_args = TensorAccessorArgs<gate_up_args.next_compile_time_args_offset()>();
    // The gate_up then down weight base addresses (one per expert) follow the accessor args
    // in the compile-time args, indexed by the runtime-selected expert id.
    constexpr uint32_t kGateUpAddrBase = down_args.next_compile_time_args_offset();
    constexpr uint32_t kDownAddrBase = kGateUpAddrBase + num_weights;

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(1);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(2);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(4);
    const uint32_t num_dests = get_arg_val<uint32_t>(5);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(6);

    // Use NoC 1 ("the other NoC") so this runs in parallel with the {0,0} sender on NoC 0.
    Noc noc(1);
    const auto input = TensorAccessor(input_args, input_addr);

    CircularBuffer cb_input(cb_input_id);

    // ---- 1. Read all activation tiles into cb_input. ----
    cb_input.reserve_back(input_num_pages);
    const uint32_t input_l1 = cb_input.get_write_ptr();
    for (uint32_t p = 0; p < input_num_pages; ++p) {
        noc.async_read(input, cb_input, input_page_size, {.page_id = p}, {.offset_bytes = p * input_page_size});
    }
    noc.async_read_barrier();

    // ---- 2. Broadcast the activation to all other cores' L1 (same cb_input address). ----
    const uint32_t total_bytes = input_page_size * input_num_pages;
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(input_l1),
        MulticastEndpoint{},
        total_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = input_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // ---- 3. Signal the other cores via the input-ready semaphore. ----
    Semaphore<> sem(sem_input_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);

    // ---- 4. Publish the activation to this core's compute kernel. ----
    cb_input.push_back(input_num_pages);

    // ---- 5. Wait for the expert-id broadcast, then run the per-expert reader loop. ----
    Semaphore<>(sem_id).wait(1);
    run_reader_loop<false>(
        noc,
        num_active,
        col_start_tile,
        i_tiles,
        k_tiles,
        gate_up_tile_bytes,
        down_slice_tiles,
        down_tile_bytes,
        act_tile_bytes,
        num_producers,
        cb_bcast_id,
        cb_weights_id,
        cb_down_w_id,
        cb_act_id,
        sem_gather_id,
        sem_bcast_id,
        mcast_start_x,
        mcast_start_y,
        mcast_end_x,
        mcast_end_y,
        num_dests,
        gate_up_args,
        kGateUpAddrBase,
        down_args,
        kDownAddrBase,
        cb_rscalar_id,
        num_weights);
}
