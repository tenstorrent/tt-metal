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

// Expert-id sender + activation-gather leader kernel (runs on core {0,0}).
//
// 1. Reads the routing-weight row and computes the selected ("hit") expert ids on
//    device (matching the host `hit = (rw.abs().sum(0) > 0).nonzero()`), compacted
//    ascending at the front of cb_bcast and padded with the sentinel.
// 2. Multicasts the ids buffer to all other compute cores' L1 (cb_bcast).
// 3. Sets + multicasts a semaphore (sem_id) to signal the other cores.
// 4. Waits for the activation broadcast and publishes it to this core's compute.
// 5. Runs the per-expert reader loop as the LEADER: fetches this core's gate_up + down
//    slices, gathers every SwiGLU core's activation chunk into the local cb_act, and
//    broadcasts the full activation back to every core for the down matmul.
//
// Compile-time args:
//   0: num_weights (total experts whose weights are provided; routing-row width)
//   1: num_active  (routing-selected experts to run)
//   2: sentinel value for unused id slots (= num_weights)
//   3: cb_routing  (L1 scratch for the routing-weight row)
//   4: cb_bcast    (L1 buffer holding the expert ids; broadcast to all cores)
//   5: routing_page_bytes
//   6: bcast_page_bytes
//   7: sem_id      (expert-ids-ready / sequencing semaphore)
//   8: cb_weights  (this core's per-expert gate_up slice)
//   9: k_tiles     (H / 32)
//   10: i_tiles    (I / 32)
//   11: gate_up_tile_bytes
//   12: sem_input_id (input-ready semaphore)
//   13: cb_input     (activation tiles, published to compute)
//   14: cb_down_w    (this core's per-expert down slice)
//   15: cb_act       (gathered activation)
//   16: down_slice_tiles
//   17: down_tile_bytes
//   18: act_tile_bytes
//   19: num_producers (number of SwiGLU cores == I/64)
//   20: sem_gather
//   21: sem_bcast
//   22: cb_rscalar
//   23+: TensorAccessorArgs(routing_weights), TensorAccessorArgs(gate_up), TensorAccessorArgs(down)
//   then: gate_up base addresses (one per expert), then down base addresses (one per expert)
//
// Runtime args:
//   0: routing_weights base address
//   1: mcast_start_x   2: mcast_start_y
//   3: mcast_end_x     4: mcast_end_y
//   5: num_dests       (number of receiver cores = total cores - 1)
//   6: col_start_tile  (this core's first output tile)
void kernel_main() {
    constexpr uint32_t num_weights = get_compile_time_arg_val(0);
    constexpr uint32_t num_active = get_compile_time_arg_val(1);
    constexpr uint32_t sentinel = get_compile_time_arg_val(2);
    constexpr uint32_t cb_routing_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(4);
    constexpr uint32_t routing_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t bcast_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(8);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t gate_up_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(15);
    constexpr uint32_t down_slice_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t down_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t num_producers = get_compile_time_arg_val(19);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(20);
    constexpr uint32_t sem_bcast_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_rscalar_id = get_compile_time_arg_val(22);

    constexpr auto routing_args = TensorAccessorArgs<23>();
    constexpr auto gate_up_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr auto down_args = TensorAccessorArgs<gate_up_args.next_compile_time_args_offset()>();
    // The gate_up then down weight base addresses (one per expert) follow the accessor args
    // in the compile-time args, indexed by the runtime-selected expert id.
    constexpr uint32_t kGateUpAddrBase = down_args.next_compile_time_args_offset();
    constexpr uint32_t kDownAddrBase = kGateUpAddrBase + num_weights;

    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(1);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(2);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(4);
    const uint32_t num_dests = get_arg_val<uint32_t>(5);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(6);

    // Pin the expert-id sender to NoC 0; the input broadcaster on {1,0} uses NoC 1.
    Noc noc(0);
    const auto routing = TensorAccessor(routing_args, routing_addr);

    CircularBuffer cb_routing(cb_routing_id);
    CircularBuffer cb_bcast(cb_bcast_id);

    // ---- 1. Read routing row + compute expert ids into cb_bcast. ----
    cb_routing.reserve_back(1);
    noc.async_read(routing, cb_routing, routing_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    cb_bcast.reserve_back(1);
    const uint32_t bcast_l1 = cb_bcast.get_write_ptr();

    CoreLocalMem<volatile uint16_t> rw(cb_routing.get_write_ptr());
    CoreLocalMem<volatile uint32_t> ids(bcast_l1);

    // ids[0..num_weights)        : compacted ascending hit ids, padded with the sentinel.
    // ids[num_weights..+num_active): each hit's routing-weight scalar as an fp32 bit pattern
    //   (bf16 value << 16), in the same hit order, for the down-output scalar broadcast.
    uint32_t n = 0;
    for (uint32_t e = 0; e < num_weights; ++e) {
        if ((rw[e] & 0x7FFF) != 0) {
            ids[n] = e;
            ids[num_weights + n] = static_cast<uint32_t>(rw[e]) << 16;
            ++n;
        }
    }
    for (uint32_t i = n; i < num_weights; ++i) {
        ids[i] = sentinel;
    }

    // ---- 2. Broadcast the ids to all other cores' L1 (same cb_bcast address). ----
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(bcast_l1),
        MulticastEndpoint{},
        bcast_page_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = bcast_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // ---- 3. Signal the other cores via the expert-ids-ready semaphore. ----
    Semaphore<> sem(sem_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);

    // ---- 4. Wait for the activation broadcast, then publish it to compute. ----
    Semaphore<>(sem_input_id).wait(1);
    publish_input(cb_input_id, k_tiles);

    // ---- 5. Two-phase reader loop (leader role): all gate_up, single gather+broadcast, all down. ----
    run_reader_loop<true>(
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
