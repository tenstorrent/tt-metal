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
//   - {1,0} multicasts the activation row into cb_input and bumps sem_input_id. With
//     INPUT_REPLICATED there is no activation broadcast: cb_input is aliased over this core's own
//     ROW_MAJOR HEIGHT_SHARDED replica, and the tiles are published without waiting on anything.
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
//   19: down_prefetch (down slices to fetch before the gather/broadcast sync)
//   20: batch        (token rows, B <= 32, sharing one activation tile row)
//   21: experts_block (experts per block; the activation block held in L1 at once)
//   22: gate_up_reserve_tiles (pages a gate_up slice reserves in cb_weights)
//   23: down_reserve_tiles    (pages a down slice reserves in cb_weights)
//   ... routing-scalar geometry / cores / reduce (indices 24-32)
//   33: split_col (first act tile of each expert that hub1 owns; == i_tiles when there is one hub)
//   34: num_hubs  (1, or 2 for the two-hub gather/broadcast)
//   35+: TensorAccessorArgs(gate_up), TensorAccessorArgs(down)
//   then: gate_up base addresses (one per expert), then down base addresses (one per expert)
//
// Runtime args:
//   0: core_index      (this core's flat grid index, x*8 + y)
//   1: hub0_noc_x      2: hub0_noc_y (the gather/broadcast hub this core's group broadcasts from)
//   3: group_mcast_start_x    4: group_mcast_start_y
//   5: group_mcast_end_x      6: group_mcast_end_y
//   7: group_num_dests
//   8: hub_role   (0 = plain, 1 = hub0, 2 = hub1)
//   9: hub1_noc_x     10: hub1_noc_y
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
    constexpr uint32_t down_prefetch = get_compile_time_arg_val(19);
    constexpr uint32_t batch = get_compile_time_arg_val(20);
    constexpr uint32_t experts_block = get_compile_time_arg_val(21);
    constexpr uint32_t gate_up_reserve_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t down_reserve_tiles = get_compile_time_arg_val(23);
    // Routing-scalar tile geometry (bf16 tile matching the input's height; width fixed at 32).
    constexpr uint32_t rscalar_tile_h = get_compile_time_arg_val(24);
    constexpr uint32_t rscalar_face_r_dim = get_compile_time_arg_val(25);
    constexpr uint32_t rscalar_num_face_rows = get_compile_time_arg_val(26);
    constexpr uint32_t rscalar_tile_bytes = get_compile_time_arg_val(27);
    constexpr uint32_t cores_per_expert = get_compile_time_arg_val(28);
    constexpr uint32_t shards_per_core = get_compile_time_arg_val(29);
    constexpr uint32_t i_shards_per_core = get_compile_time_arg_val(30);
    constexpr uint32_t num_expert_groups = get_compile_time_arg_val(31);
    constexpr uint32_t sem_reduce_id = get_compile_time_arg_val(32);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(33);
    constexpr uint32_t split_col = get_compile_time_arg_val(34);
    constexpr uint32_t num_hubs = get_compile_time_arg_val(35);

    constexpr auto gate_up_args = TensorAccessorArgs<36>();
    constexpr auto down_args = TensorAccessorArgs<gate_up_args.next_compile_time_args_offset()>();
    // The gate_up then down weight base addresses (one per expert) follow the accessor args
    // in the compile-time args, indexed by the runtime-selected expert id.
    constexpr uint32_t kGateUpAddrBase = down_args.next_compile_time_args_offset();
    constexpr uint32_t kDownAddrBase = kGateUpAddrBase + num_weights;

    const uint32_t core_index = get_arg_val<uint32_t>(0);
    const uint32_t hub0_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t hub0_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t group_mcast_start_x = get_arg_val<uint32_t>(3);
    const uint32_t group_mcast_start_y = get_arg_val<uint32_t>(4);
    const uint32_t group_mcast_end_x = get_arg_val<uint32_t>(5);
    const uint32_t group_mcast_end_y = get_arg_val<uint32_t>(6);
    const uint32_t group_num_dests = get_arg_val<uint32_t>(7);
    const uint32_t hub_role = get_arg_val<uint32_t>(8);
    const uint32_t hub1_noc_x = get_arg_val<uint32_t>(9);
    const uint32_t hub1_noc_y = get_arg_val<uint32_t>(10);

    const ActGatherConfig gather{hub_role, hub0_noc_x, hub0_noc_y, hub1_noc_x, hub1_noc_y, split_col, num_hubs};

    // Activation: publish it to the compute kernel.
#ifdef INPUT_REPLICATED
    // Already in this core's L1 as a ROW_MAJOR HEIGHT_SHARDED replica, with cb_input aliased over
    // its shard -- nothing was broadcast, so there is nothing to wait for.
    publish_input(cb_input_id, k_tiles);
    (void)sem_input_id;
#else
    // Arrived via the {1,0} input sender's multicast.
    Semaphore<>(sem_input_id).wait(1);
    publish_input(cb_input_id, k_tiles);
#endif

    // Expert ids arrived, then run the per-expert reader loop.
    Semaphore<>(sem_id).wait(1);
    Noc noc;
    run_reader_loop(
        noc,
        num_active,
        core_index,
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
        group_mcast_start_x,
        group_mcast_start_y,
        group_mcast_end_x,
        group_mcast_end_y,
        group_num_dests,
        gate_up_args,
        kGateUpAddrBase,
        down_args,
        kDownAddrBase,
        cb_rscalar_id,
        num_weights,
        down_prefetch,
        batch,
        experts_block,
        gate_up_reserve_tiles,
        down_reserve_tiles,
        rscalar_tile_h,
        rscalar_face_r_dim,
        rscalar_num_face_rows,
        rscalar_tile_bytes,
        cores_per_expert,
        shards_per_core,
        i_shards_per_core,
        num_expert_groups,
        gather,
        sem_reduce_id,
        cb_reduce_id);
}
