// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "fetch_gate_up.h"

// Reader kernel (runs on every compute core except the two senders {0,0} and {1,0}).
//
// Waits for both initial broadcasts to land in this core's L1:
//   - {1,0} multicasts the activation row into cb_input and bumps sem_input_id.
//   - {0,0} multicasts the expert ids into cb_bcast and bumps sem_id.
// It then publishes the activation to the compute kernel (cb_input) and runs the
// per-expert reader loop as a (non-leader) receiver: fetches this core's gate_up + down
// slices and receives the broadcast activation for the down matmul.
//
// Compile-time args:
//   0: sem_id        (expert-ids-ready / sequencing semaphore)
//   1: sem_input_id  (input-ready semaphore)
//   2: num_active    (routing-selected experts to run)
//   3: cb_input      (activation tiles, published to compute)
//   4: cb_weights    (this core's per-expert gate_up slice)
//   5: k_tiles       (H / 32)
//   6: i_tiles       (I / 32)
//   7: gate_up_tile_bytes
//   8: cb_bcast      (broadcast hit-expert ids, read by the weight fetch)
//   9: cb_down_w     (this core's per-expert down slice)
//   10: cb_act       (gathered activation)
//   11: down_slice_tiles
//   12: down_tile_bytes
//   13: act_tile_bytes
//   14: num_producers
//   15: sem_gather
//   16: sem_bcast
//   17: num_weights
//   18: cb_rscalar
//   19+: TensorAccessorArgs(gate_up), TensorAccessorArgs(down)
//   then: gate_up base addresses (one per expert), then down base addresses (one per expert)
//
// Runtime args:
//   0: col_start_tile  (this core's first output tile)
void kernel_main() {
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_active = get_compile_time_arg_val(2);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t gate_up_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(10);
    constexpr uint32_t down_slice_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t down_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t num_producers = get_compile_time_arg_val(14);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(15);
    constexpr uint32_t sem_bcast_id = get_compile_time_arg_val(16);
    constexpr uint32_t num_weights = get_compile_time_arg_val(17);
    constexpr uint32_t cb_rscalar_id = get_compile_time_arg_val(18);

    constexpr auto gate_up_args = TensorAccessorArgs<19>();
    constexpr auto down_args = TensorAccessorArgs<gate_up_args.next_compile_time_args_offset()>();
    // The gate_up then down weight base addresses (one per expert) follow the accessor args
    // in the compile-time args, indexed by the runtime-selected expert id.
    constexpr uint32_t kGateUpAddrBase = down_args.next_compile_time_args_offset();
    constexpr uint32_t kDownAddrBase = kGateUpAddrBase + num_weights;

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);

    // Activation arrived via multicast: publish it to the compute kernel.
    Semaphore<>(sem_input_id).wait(1);
    publish_input(cb_input_id, k_tiles);

    // Expert ids arrived, then run the per-expert reader loop.
    Semaphore<>(sem_id).wait(1);
    Noc noc;
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
        /*mcast_start_x=*/0,
        /*mcast_start_y=*/0,
        /*mcast_end_x=*/0,
        /*mcast_end_y=*/0,
        /*num_dests=*/0,
        gate_up_args,
        kGateUpAddrBase,
        down_args,
        kDownAddrBase,
        cb_rscalar_id,
        num_weights);
}
