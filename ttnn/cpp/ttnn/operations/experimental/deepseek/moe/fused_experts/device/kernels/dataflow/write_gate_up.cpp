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
//      swiglu_tiles-tile SwiGLU activation slice (cb_out) into core {0,0}'s cb_act slot for the
//      expert's block, at tile offset (j*i_tiles + core_index*swiglu_tiles) for the block's expert
//      j -- a single NoC write -- then bumps the leader's gather semaphore. Once a block's chunks
//      from all SwiGLU cores have landed, {0,0} broadcasts that slot back to every core in one shot
//      (see the reader kernels). Writing into the slot needs no handshake: this core cannot have
//      produced a block's activation until its compute consumed the previous broadcast, which the
//      leader only sent after reserving the slot being written here.
//
//   PHASE 2 -- WRITE OUTPUT (all cores): the compute kernel has accumulated the routing-weighted
//      down outputs of all experts into a single tile row (all B tokens), so this core writes its
//      2-tile slice of that row (cb_down_out) once to the [1, B, H] DRAM output. No semaphore is
//      needed.
//
// Output is TILE [1, B, H] (the B token rows padded to a 32-row tile), so in tiles it is
// [1, 1, h_tiles] with h_tiles == H/32 -- one tile row whatever B is, which is why nothing in this
// kernel depends on the batch. Core idx owns the 2 H-dim tiles [idx*2, idx*2 + 1],
// written to output pages idx*2 + {0, 1}. Note the SwiGLU I-dim split (swiglu_tiles per core)
// is independent of this H-dim split, so the two offsets are computed separately.
//
// Compile-time args:
//   0: num_active     (routing-selected experts to run)
//   1: i_tiles        (I/32; activation stride between experts in the gathered block)
//   2: cb_out         (this core's SwiGLU activation tiles per expert)
//   3: cb_down_out    (this core's 2 accumulated output tiles)
//   4: cb_act         (gathered activation; used only to locate the leader's L1 address)
//   5: act_tile_bytes (bytes per activation tile)
//   6: out_tile_bytes (bytes per output tile)
//   7: sem_gather     (leader's gather semaphore)
//   8: num_producers  (number of SwiGLU cores; SwiGLU-core guard for the gather scatter)
//   9: experts_block  (experts per block; cb_act holds one block per slot)
//   10+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: output base address
//   1: core_index (this core's flat grid index, x*8 + y)
//   2: leader_noc_x   3: leader_noc_y (core {0,0} NoC coords for this writer's NoC)
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

    constexpr auto out_args = TensorAccessorArgs<10>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_index = get_arg_val<uint32_t>(1);
    const uint32_t leader_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t leader_noc_y = get_arg_val<uint32_t>(3);

    constexpr uint32_t kOutTiles = 2;
    constexpr uint32_t swiglu_tiles = i_tiles / num_producers;
    const bool swiglu_core = core_index < num_producers;
    const uint32_t swiglu_col_start = core_index * swiglu_tiles;
    const uint32_t out_col_start = core_index * kOutTiles;

    Noc noc;
    const auto out = TensorAccessor(out_args, out_addr);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_down_out(cb_down_out_id);
    CircularBuffer cb_act(cb_act_id);

    // cb_act is allocated identically on every core, so the leader's cb_act base equals this core's
    // cb_act BASE address -- read here, before anything can have advanced the write pointer. (Nothing
    // can: cb_act is first pushed when the leader publishes block 0, which cannot happen until every
    // writer has scattered into it, and each writer reads this first.) Slot addresses are then derived
    // arithmetically rather than from the pointer, whose position is this core's own reader's business.
    const uint32_t leader_act_base = cb_act.get_write_ptr();
    constexpr uint32_t kNumBlocks = (num_active + experts_block - 1) / experts_block;
    const uint32_t block_act_tiles = experts_block * i_tiles;
    // One slot when everything fits in a single block, two when blocks alternate through cb_act.
    const uint32_t act_slots = kNumBlocks > 1 ? 2 : 1;

    Semaphore<> sem_gather(sem_gather_id);

    // ---- Phase 1: scatter every expert's SwiGLU slice into the leader's cb_act. ----
    if (swiglu_core) {
        for (uint32_t blk = 0; blk < kNumBlocks; ++blk) {
            const uint32_t first_expert = blk * experts_block;
            const uint32_t remaining = num_active - first_expert;
            const uint32_t block_experts = remaining < experts_block ? remaining : experts_block;
            const uint32_t slot_base = leader_act_base + (blk % act_slots) * block_act_tiles * act_tile_bytes;

            for (uint32_t j = 0; j < block_experts; ++j) {
                cb_out.wait_front(swiglu_tiles);
                const uint32_t leader_addr = slot_base + (j * i_tiles + swiglu_col_start) * act_tile_bytes;
                noc.async_write(
                    cb_out,
                    UnicastEndpoint{},
                    swiglu_tiles * act_tile_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = leader_noc_x, .noc_y = leader_noc_y, .addr = leader_addr});
                noc.async_write_barrier();
                sem_gather.up(noc, leader_noc_x, leader_noc_y, 1);
                cb_out.pop_front(swiglu_tiles);
            }
        }
    }

    // ---- Phase 2: write this core's slice of the single accumulated output row to DRAM. ----
    // The compute kernel has already summed routing_w[e] * down_e over all experts into
    // cb_down_out, so the writer drains it once into output pages idx*2 + {0, 1}.
    cb_down_out.wait_front(kOutTiles);
    for (uint32_t t = 0; t < kOutTiles; ++t) {
        const uint32_t page = out_col_start + t;
        noc.async_write(cb_down_out, out, out_tile_bytes, {.offset_bytes = t * out_tile_bytes}, {.page_id = page});
    }
    noc.async_write_barrier();
    cb_down_out.pop_front(kOutTiles);
}
