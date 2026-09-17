// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Writer / gather-scatter kernel (runs on every compute core), structured in two phases:
//
//   PHASE 1 -- GATHER (SwiGLU cores only): for every expert, copies this core's
//      swiglu_tiles-tile SwiGLU activation slice (cb_out) into the cb_act slot of the hub that
//      owns that slice, at tile offset (j*i_tiles + col_start) -- a single NoC write -- then bumps
//      BOTH hubs' gather semaphores. The I dim is split at `split_col`: a slice starting below it
//      goes to hub0, at or above it to hub1. Hub0/hub1 are the two corners of the gather's
//      multicast rectangle -- {0,0} and {GRID_X-1, GRID_Y-1} on the 64-core path (col_start is
//      core_index*swiglu_tiles), and the first/last core of this expert's 16-core group on the
//      6-expert / 96-core path (col_start is local_idx*i_shards_per_core, the core's I-tiles
//      written together -- 1 at TP I=512, 4 at I=2048).
//
//   PHASE 2 -- WRITE OUTPUT:
//      64-core path: the compute kernel has accumulated every expert into cb_down_out, so this
//      core writes its 2-tile H-slice to DRAM.
//      6-expert path: each group holds one expert's H-slice. Non-root groups unicast their
//      shards_per_core*2 tiles to the matching core in group 0; group 0's compute reduces them
//      and this writer then drains the sum to DRAM.
//
// Compile-time args:
//   0: num_active     (routing-selected experts to run)
//   1: i_tiles        (I/32; activation stride between experts in the gathered block)
//   2: cb_out         (this core's SwiGLU activation tiles per expert)
//   3: cb_down_out    (this core's accumulated output tiles)
//   4: cb_act         (gathered activation; used only to locate the leader's L1 address)
//   5: act_tile_bytes (bytes per activation tile)
//   6: out_tile_bytes (bytes per output tile)
//   7: sem_gather     (leader's gather semaphore)
//   8: num_producers  (number of SwiGLU cores in the gather group)
//   9: experts_block  (experts per block; cb_act holds one block per slot)
//   10: cores_per_expert
//   11: shards_per_core (down/H shards this core covers)
//   12: i_shards_per_core (gate_up/I shards this core covers)
//   13: num_expert_groups (1 = 64-core path, 6 = 96-core path)
//   14: cb_reduce
//   15: sem_reduce
//   16: dst_tiles_per_page (compute tiles packed into one output TensorAccessor page;
//       TILE == 1, ROW_MAJOR stick/shard may hold several 1x32 faces)
//   17: split_col (first act tile of each expert that hub1 owns; == i_tiles when there is one hub)
//   18: num_hubs  (1, or 2 for the two-hub gather/broadcast)
//   19+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: output base address
//   1: core_index (this core's flat grid index, x*GRID_Y + y)
//   2: hub0_noc_x     3: hub0_noc_y (this core's group's hub0, the low-I gather target)
//   4: reduce_noc_x   5: reduce_noc_y (group-0 counterpart, 6-expert path only)
//   6: hub1_noc_x     7: hub1_noc_y (this core's group's hub1, the high-I gather target)
//
// A producer scatters its I slice to whichever hub owns it (the slice's first column decides: below
// `split_col` goes to hub0, at or above it to hub1) and bumps BOTH hubs' gather semaphores. Bumping
// the hub that did not receive the chunk matters: that bump is the evidence the other hub needs that
// this core's compute has moved past the previous block, and it is the only such evidence for a core
// whose chunk that hub never gathers.
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_down_out_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(4);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(7);
    constexpr uint32_t num_producers = get_compile_time_arg_val(8);
    constexpr uint32_t experts_block = get_compile_time_arg_val(9);
    constexpr uint32_t cores_per_expert = get_compile_time_arg_val(10);
    constexpr uint32_t shards_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t i_shards_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t num_expert_groups = get_compile_time_arg_val(13);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_reduce_id = get_compile_time_arg_val(15);
    constexpr uint32_t dst_tiles_per_page = get_compile_time_arg_val(16);
    constexpr uint32_t split_col = get_compile_time_arg_val(17);
    constexpr uint32_t num_hubs = get_compile_time_arg_val(18);

    constexpr auto out_args = TensorAccessorArgs<19>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_index = get_arg_val<uint32_t>(1);
    const uint32_t hub0_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t hub0_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t reduce_noc_x = get_arg_val<uint32_t>(4);
    const uint32_t reduce_noc_y = get_arg_val<uint32_t>(5);
    const uint32_t hub1_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t hub1_noc_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t kOutTiles = 2;
    constexpr uint32_t swiglu_tiles =
        num_expert_groups > 1 ? (i_tiles / (cores_per_expert * i_shards_per_core)) : (i_tiles / num_producers);
    const uint32_t local_idx = core_index % cores_per_expert;
    const uint32_t expert_group = core_index / cores_per_expert;
    const bool swiglu_core = local_idx < num_producers;
    const uint32_t swiglu_col_start =
        num_expert_groups > 1 ? local_idx * i_shards_per_core * swiglu_tiles : core_index * swiglu_tiles;
    const uint32_t out_col_start =
        num_expert_groups > 1 ? local_idx * shards_per_core * kOutTiles : core_index * kOutTiles;
    // This core's chunk belongs to hub1 if it starts at or past the I split; otherwise to hub0. With
    // one hub `split_col == i_tiles`, so every chunk goes to hub0.
    const bool chunk_is_hub1 = swiglu_col_start >= split_col;
    const uint32_t chunk_hub_x = chunk_is_hub1 ? hub1_noc_x : hub0_noc_x;
    const uint32_t chunk_hub_y = chunk_is_hub1 ? hub1_noc_y : hub0_noc_y;

    Noc noc;
    const auto out = TensorAccessor(out_args, out_addr);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_down_out(cb_down_out_id);
    CircularBuffer cb_act(cb_act_id);
    CircularBuffer cb_reduce(cb_reduce_id);

    // cb_act / cb_reduce are allocated identically on every core, so every hub's staging region
    // shares this core's BASE address -- read here, before anything can have advanced the write
    // pointer.
    const uint32_t act_slot_base = cb_act.get_write_ptr();
    const uint32_t reduce_base = cb_reduce.get_write_ptr();
    constexpr uint32_t kNumBlocks = (num_active + experts_block - 1) / experts_block;
    const uint32_t block_act_tiles = experts_block * i_tiles;
    const uint32_t act_slots = kNumBlocks > 1 ? 2 : 1;

    Semaphore<> sem_gather(sem_gather_id);

    // The gather bump goes to BOTH hubs: the owning hub needs it to know the chunk landed, and the
    // other hub needs it as the only evidence that this core is done with the previous block's slot
    // (that hub never gathers this core's chunk, so it has no scatter of its own to observe).
    auto bump_gather = [&]() {
        sem_gather.up(noc, hub0_noc_x, hub0_noc_y, 1);
        if constexpr (num_hubs > 1) {
            sem_gather.up(noc, hub1_noc_x, hub1_noc_y, 1);
        }
    };

    if constexpr (num_expert_groups > 1) {
        const uint32_t scatter_tiles = i_shards_per_core * swiglu_tiles;
        if (swiglu_core) {
            cb_out.wait_front(scatter_tiles);
            noc.async_write(
                cb_out,
                UnicastEndpoint{},
                scatter_tiles * act_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = chunk_hub_x,
                 .noc_y = chunk_hub_y,
                 .addr = act_slot_base + swiglu_col_start * act_tile_bytes});
            noc.async_write_barrier();
            bump_gather();
            cb_out.pop_front(scatter_tiles);
        }

        constexpr uint32_t kLocalTiles = shards_per_core * kOutTiles;
        cb_down_out.wait_front(kLocalTiles);
        if (expert_group == 0) {
            for (uint32_t t = 0; t < kLocalTiles; ++t) {
                const uint32_t global_tile = out_col_start + t;
                noc.async_write(
                    cb_down_out,
                    out,
                    out_tile_bytes,
                    {.offset_bytes = t * out_tile_bytes},
                    {.page_id = global_tile / dst_tiles_per_page,
                     .offset_bytes = (global_tile % dst_tiles_per_page) * out_tile_bytes});
            }
            noc.async_write_barrier();
        } else {
            const uint32_t slot = (expert_group - 1u) * kLocalTiles * out_tile_bytes;
            noc.async_write(
                cb_down_out,
                UnicastEndpoint{},
                kLocalTiles * out_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = reduce_noc_x, .noc_y = reduce_noc_y, .addr = reduce_base + slot});
            noc.async_write_barrier();
            Semaphore<>(sem_reduce_id).up(noc, reduce_noc_x, reduce_noc_y, 1);
        }
        cb_down_out.pop_front(kLocalTiles);
    } else {
        if (swiglu_core) {
            for (uint32_t blk = 0; blk < kNumBlocks; ++blk) {
                const uint32_t first_expert = blk * experts_block;
                const uint32_t remaining = num_active - first_expert;
                const uint32_t block_experts = remaining < experts_block ? remaining : experts_block;
                const uint32_t slot_base = act_slot_base + (blk % act_slots) * block_act_tiles * act_tile_bytes;

                for (uint32_t j = 0; j < block_experts; ++j) {
                    cb_out.wait_front(swiglu_tiles);
                    const uint32_t hub_addr = slot_base + (j * i_tiles + swiglu_col_start) * act_tile_bytes;
                    noc.async_write(
                        cb_out,
                        UnicastEndpoint{},
                        swiglu_tiles * act_tile_bytes,
                        {.offset_bytes = 0},
                        {.noc_x = chunk_hub_x, .noc_y = chunk_hub_y, .addr = hub_addr});
                    noc.async_write_barrier();
                    bump_gather();
                    cb_out.pop_front(swiglu_tiles);
                }
            }
        }

        // The compute kernel has already summed routing_w[e] * down_e over all experts into
        // cb_down_out, so the writer drains it once into output pages idx*2 + {0, 1}.
        cb_down_out.wait_front(kOutTiles);
        for (uint32_t t = 0; t < kOutTiles; ++t) {
            const uint32_t global_tile = out_col_start + t;
            noc.async_write(
                cb_down_out,
                out,
                out_tile_bytes,
                {.offset_bytes = t * out_tile_bytes},
                {.page_id = global_tile / dst_tiles_per_page,
                 .offset_bytes = (global_tile % dst_tiles_per_page) * out_tile_bytes});
        }
        noc.async_write_barrier();
        cb_down_out.pop_front(kOutTiles);
    }
}
