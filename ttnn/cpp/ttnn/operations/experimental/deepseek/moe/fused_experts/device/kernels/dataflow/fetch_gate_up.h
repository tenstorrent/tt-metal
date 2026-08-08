// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "api/compile_time_args.h"
#include "tools/profiler/kernel_profiler.hpp"

// Shared dataflow helpers for the fused-experts pipeline (used by every DM kernel).
//
// The selected experts are processed in BLOCKS of `experts_block`, and the pipeline below runs once
// per block: only one block's activations are resident in L1, so a batch may select far more distinct
// experts than L1 could hold at once (32 tokens at top_k 6 select up to 192). Blocking changes no
// arithmetic and no DRAM traffic -- each expert is still fetched exactly once -- at the cost of one
// gather/broadcast synchronization per block. A single block reproduces the original pipeline.
//
// PER-EXPERT PIPELINE (run by the reader on every core, in lock-step across the chip):
//   1. gate_up matmul + SwiGLU produces, on each of the I/64 SwiGLU cores, a 2-tile
//      (64-column) slice of the activation act[B, I] (its I-columns [idx*64, idx*64+64)). The B
//      tokens of the batch are the rows of that slice, so they ride along at no extra cost.
//   2. GATHER: each SwiGLU core's writer copies its 2 act tiles into core {0,0}'s cb_act
//      slot for the current block (a single NoC write to the leader) and bumps the leader's
//      gather semaphore. After all chunks land, that slot holds the block's act[B, I] per
//      expert (i_tiles == I/32 tiles each, in K order).
//   3. BROADCAST: {0,0} multicasts the whole slot to every other core's cb_act (same L1
//      address) and publishes the block's number in the broadcast semaphore. Now every core has
//      the block's activations.
//   4. DOWN matmul: each of the 64 cores multiplies the full act[B, I] by its own down
//      weight shard ([I, H/64] -> down_slice_tiles tiles) to produce its 2-tile (64-column)
//      slice of the output row[B, H]; the compute kernel scales it by each token's routing
//      weight for that expert and accumulates across experts into the [1, B, H] DRAM output.
//
// DRAM BANDWIDTH OPTIMIZATION (see tech_reports/Saturating_DRAM_bandwidth):
// The weight fetch is the DRAM-bound path of this op. Each expert's per-core weight slice
// is one NoC read. Two techniques from the saturating-DRAM-bandwidth work apply here:
//   1. Sharded tensors in DRAM: the gate_up and down weights are ND-sharded so each core
//      reads only its own partition from its assigned bank (no round-robin interleaved
//      access that would cause NoC congestion) -- already in place.
//   2. Double-buffered weight CBs: the per-expert weight slice is double-buffered so the
//      reader can hold one expert's slice ready in L1 while compute consumes the previous
//      expert's, overlapping data movement with computation (the tech report's
//      "In0 and in1 shards are also double buffered, to overlap the data movement with
//      computation"). The gate_up weight CB is now sized for two slots (it previously held
//      a single slice), and the down weight CB was already double-buffered; both reuse the
//      same CB index across the two phases (gate_up is dead during the down phase).

// The activation row is delivered into every core's cb_input L1 region by the input
// broadcaster's multicast (receivers) or by a direct DRAM read (the broadcaster itself).
// Advancing cb_input by k_tiles pages publishes it to the matmul compute kernel.
inline void publish_input(uint32_t cb_input_id, uint32_t k_tiles) {
    CircularBuffer cb_input(cb_input_id);
    cb_input.reserve_back(k_tiles);
    cb_input.push_back(k_tiles);
}

// gate_up weight layout: each SwiGLU core owns an `swiglu_tiles` slice of the SwiGLU output I
// dim (I/32 tiles spread over the 64 cores), needing the matching gate columns and paired up
// columns of the [K, 2I] weight. The host permutes the weight into per-core [gate | up]
// blocks so each DRAM shard is this core's [K, 64*swiglu_tiles] slice (the first half of the
// tile cols is gate, the second up), read in one NoC read. Shard id == this core's grid index.

// Read this core's weight slice for the i-th selected ("hit") expert into a double-buffered
// weight CB. `i` indexes the deduplicated union of the batch's selections, so this runs once per
// distinct expert, never once per (token, expert) pair.
//
// The CB's reserve_back/push_back provide compute-side flow control: with two
// slots the reader can hold one expert's data ready while compute consumes the previous
// expert's, decoupling the reader from compute (see tech_reports/Saturating_DRAM_bandwidth --
// "double buffered, to overlap the data movement with computation").
//
// `cb_weights_id` must be a CB with total_size >= 2 * reserve_tiles * page_size.
// `ct_w_addr_base` is the compile-time-args offset of the per-expert weight base addresses.
//
// `reserve_tiles` is how far the CB pointer advances, which is `slice_tiles` unless the host padded
// it so that the gate_up and down phases advance by the same stride (they share this CB and alternate
// in it once experts are blocked; see the program factory). The read itself is always slice_tiles,
// leaving the pad at the end of the slot untouched -- and unread, since the consumer indexes tiles
// from the front of the slot.
template <typename WeightArgs>
inline void fetch_weight_one(
    const Noc& noc,
    uint32_t cb_bcast_id,
    uint32_t cb_weights_id,
    uint32_t i,
    uint32_t slice_tiles,
    uint32_t reserve_tiles,
    uint32_t tile_bytes,
    uint32_t shard_id,
    const WeightArgs& weight_args,
    uint32_t ct_w_addr_base) {
    const uint32_t slice_bytes = slice_tiles * tile_bytes;

    CircularBuffer cb_bcast(cb_bcast_id);
    CoreLocalMem<volatile uint32_t> ids(cb_bcast.get_write_ptr());
    const uint32_t expert = ids[i];
    // Weight base addresses live in the compile-time args (in expert-id order); index the
    // resident kernel_compile_time_args array by the runtime-selected expert id directly.
    const uint32_t w_addr = kernel_compile_time_args[ct_w_addr_base + expert];
    const auto w = TensorAccessor(weight_args, w_addr);

    CircularBuffer cb_weights(cb_weights_id);
    cb_weights.reserve_back(reserve_tiles);
    ShardView w_shard(w);
    noc.async_read(w_shard, cb_weights, slice_bytes, {.shard_id = shard_id}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_weights.push_back(reserve_tiles);
}

// Leader ({0,0}) side of ONE BLOCK's activation gather + broadcast.
//
// By the time this runs, every SwiGLU core has produced the block's activation slices (phase 1) and
// its writer has scattered them into `act_l1`, the leader's cb_act slot for this block -- the block's
// expert j chunk for core idx at tile offset (j*i_tiles + idx*swiglu_tiles). The slot therefore holds
// the block's [n, B, I] activations once `gather_target` bumps have landed (num_producers per expert,
// counted cumulatively across blocks, plus one ack per block from any core that has no chunk to
// send). The leader then multicasts the whole slot to every other core in one shot. Within a block
// cb_act is never reused, so no per-expert back-pressure is needed.
//
// `next_block_act_tiles` is the next block's size, or 0 if this is the last block (or the only one).
// Reserving that slot *before* the broadcast is what makes blocking safe: it returns only once this
// core's compute has released the slot the next block will reuse, so a core that has seen this
// broadcast may scatter into the leader's next slot without any further handshake.
inline void leader_gather_broadcast_block(
    const Noc& noc,
    uint32_t cb_act_id,
    uint32_t act_l1,
    uint32_t block_act_tiles,
    uint32_t next_block_act_tiles,
    uint32_t act_tile_bytes,
    uint32_t gather_target,
    uint32_t sem_gather_id,
    uint32_t sem_bcast_id,
    uint32_t blocks_done,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y,
    uint32_t num_dests) {
    CircularBuffer cb_act(cb_act_id);

    // This block's synchronization point: wait for each of its experts' chunks from every SwiGLU core.
    Semaphore<>(sem_gather_id).wait_min(gather_target);

    // Publish the block to the local compute kernel first: the slot is already complete, so the down
    // matmul can start while the multicast below is still in flight.
    cb_act.push_back(block_act_tiles);

    // Claim the next block's slot before announcing this one (see above). This is also what leaves
    // the write pointer on that slot, which the caller reads back as the next block's act_l1.
    if (next_block_act_tiles > 0) {
        cb_act.reserve_back(next_block_act_tiles);
    }

    // Broadcast the block's activations to every other core's cb_act (same L1 address everywhere).
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(act_l1),
        MulticastEndpoint{},
        block_act_tiles * act_tile_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = act_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // The semaphore carries the number of blocks broadcast so far rather than a flag, so it never has
    // to be reset between blocks -- a reset would race with a core still reading the previous value.
    Semaphore<> sem(sem_bcast_id);
    sem.set(blocks_done);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);
}

// Non-leader side: wait for the leader's broadcast of one block of activations, then publish it
// (already resident, written straight into this core's cb_act by the multicast) to the local compute.
//
// The slot is claimed first, so it is only handed to the leader once compute has released whatever
// block previously occupied it. For a core that scatters, that release is already implied by the
// scatter itself -- it cannot happen until its compute has moved past the previous block -- but a
// core with no slice of the I dim sends no chunk, so it acks the gather here instead. Without that
// ack the leader would have no evidence that such a core is done with the slot it is about to
// overwrite.
inline void receiver_recv_act_block(
    const Noc& noc,
    uint32_t cb_act_id,
    uint32_t block_act_tiles,
    uint32_t sem_bcast_id,
    uint32_t blocks_done,
    bool send_slot_free_ack,
    uint32_t sem_gather_id,
    uint32_t leader_noc_x,
    uint32_t leader_noc_y) {
    CircularBuffer cb_act(cb_act_id);
    cb_act.reserve_back(block_act_tiles);
    if (send_slot_free_ack) {
        Semaphore<>(sem_gather_id).up(noc, leader_noc_x, leader_noc_y, 1);
    }
    Semaphore<>(sem_bcast_id).wait_min(blocks_done);
    cb_act.push_back(block_act_tiles);
}

// Build one bf16 routing-weight tile per expert of the current block (experts
// [first_expert, first_expert + count)) from the per-token weights the leader appended to cb_bcast
// (at index weight_base + e*batch + b, as fp32 bit patterns). Only a block's tiles are held at a
// time; the source weights stay resident in cb_bcast, so each block just refills the same tiles.
//
// Tile row b holds expert e's routing weight for token b, splatted across the row, so the
// down-output stage applies every token's own weight with a single elementwise multiply. That is
// what lets one fetch + one matmul of a shared expert serve all the tokens that selected it: the
// tokens are separated only here, by their weight, and a token that did not select the expert has
// weight 0. Rows past `batch` are zeroed, so the tile-padding rows of the output stay zero.
//
// A tile is 4 16x16 faces (faces 0,1 = rows 0-15, faces 2,3 = rows 16-31), so row r spans 16
// elements at row r%16 of face (r/16)*2 and the same offset of the next face. Every store is a
// 32-bit write of two bf16 lanes: this loop runs on every core ahead of the DRAM-bound phases, so
// halving its store count is pure critical-path savings, and writing every row (rather than zeroing
// the tile first and then filling it) touches each word exactly once.
inline void build_routing_scalars(
    uint32_t cb_bcast_id,
    uint32_t cb_rscalar_id,
    uint32_t first_expert,
    uint32_t count,
    uint32_t weight_base,
    uint32_t batch) {
    CircularBuffer cb_bcast(cb_bcast_id);
    CoreLocalMem<volatile uint32_t> bcast(cb_bcast.get_write_ptr());

    CircularBuffer cb_rscalar(cb_rscalar_id);
    cb_rscalar.reserve_back(count);
    const uint32_t rscalar_l1 = cb_rscalar.get_write_ptr();

    constexpr uint32_t kTileBytes = 1024 * 2;  // 32x32 bf16 tile
    constexpr uint32_t kFaceWords = 256 / 2;   // 16x16 bf16 face, in 32-bit words
    constexpr uint32_t kRowWords = 16 / 2;     // one face row (16 bf16), in 32-bit words
    for (uint32_t e = 0; e < count; ++e) {
        CoreLocalMem<volatile uint32_t> tile(rscalar_l1 + e * kTileBytes);
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t w_pair = 0;
            if (r < batch) {
                // fp32 bit pattern -> bf16 == high 16 bits, packed into both lanes of the store.
                const uint32_t w_bf16 = bcast[weight_base + (first_expert + e) * batch + r] >> 16;
                w_pair = (w_bf16 << 16) | w_bf16;
            }
            const uint32_t face_pair = (r >> 4) * 2 * kFaceWords;  // rows 0-15 -> faces 0,1; 16-31 -> 2,3
            const uint32_t row_off = (r & 15) * kRowWords;
            for (uint32_t j = 0; j < kRowWords; ++j) {
                tile[face_pair + row_off + j] = w_pair;
                tile[face_pair + kFaceWords + row_off + j] = w_pair;
            }
        }
    }
    cb_rscalar.push_back(count);
}

// Per-core reader loop shared by all DM reader kernels: one pass per BLOCK of experts, each pass
// two phases around a single synchronization.
//
// Both phases loop over HIT INDEX, not over token: the id list from the leader is the deduplicated
// union of the batch's selections, so each distinct expert's gate_up and down slices are fetched
// exactly once here no matter how many of the B tokens selected it, and the single matmul that
// consumes them covers the whole token tile row. Batching is therefore pure DRAM-traffic
// amortization -- the fetch cost is set by the number of distinct experts, not by the token count.
// Blocking does not change that: each expert belongs to exactly one block.
//   Phase 1: fetch this core's gate_up slice for the block's experts (SwiGLU cores only). The
//            compute kernel produces their SwiGLU activations, which the writer scatters to the
//            leader.
//   Sync:    the leader gathers the block's activations and broadcasts the whole block to every
//            core in one shot; everyone else waits for it -- and, while waiting, prefetches up to
//            `down_prefetch` of the block's down slices so DRAM keeps streaming through the barrier.
//   Phase 2: fetch the block's remaining down slices. The compute kernel runs the down matmul for
//            the block's experts against the now-resident activations and accumulates.
template <bool IsLeader, typename GateUpArgs, typename DownArgs>
inline void run_reader_loop(
    const Noc& noc,
    uint32_t num_active,
    uint32_t core_index,
    uint32_t i_tiles,
    uint32_t k_tiles,
    uint32_t gate_up_tile_bytes,
    uint32_t down_slice_tiles,
    uint32_t down_tile_bytes,
    uint32_t act_tile_bytes,
    uint32_t num_producers,
    uint32_t cb_bcast_id,
    uint32_t cb_weights_id,
    uint32_t cb_down_w_id,
    uint32_t cb_act_id,
    uint32_t sem_gather_id,
    uint32_t sem_bcast_id,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y,
    uint32_t num_dests,
    const GateUpArgs& gate_up_args,
    uint32_t ct_gu_addr_base,
    const DownArgs& down_args,
    uint32_t ct_down_addr_base,
    uint32_t cb_rscalar_id,
    uint32_t weight_base,
    uint32_t down_prefetch,
    uint32_t batch,
    uint32_t experts_block,
    uint32_t gate_up_reserve_tiles,
    uint32_t down_reserve_tiles,
    uint32_t leader_noc_x,
    uint32_t leader_noc_y) {
    // Both the gate_up and the down DRAM shards for this core are shard `core_index`; the
    // first `num_producers` cores also own a slice of the SwiGLU I dim.
    const bool swiglu_core = core_index < num_producers;
    const uint32_t shard_id = core_index;
    const uint32_t swiglu_tiles = i_tiles / num_producers;
    const uint32_t gate_up_slice_tiles = k_tiles * (2u * swiglu_tiles);

    const uint32_t num_blocks = (num_active + experts_block - 1u) / experts_block;
    // Cores that own no slice of the I dim produce no activation chunk, so they have to ack the
    // gather once per block instead (see receiver_recv_act_block). Only the leader needs the count,
    // and only the leader is given the grid size (as the multicast destination count).
    const uint32_t slot_free_acks_per_block = IsLeader ? (num_dests + 1u - num_producers) : 0u;

    auto fetch_down = [&](uint32_t e) {
        fetch_weight_one(
            noc,
            cb_bcast_id,
            cb_down_w_id,
            e,
            down_slice_tiles,
            down_reserve_tiles,
            down_tile_bytes,
            shard_id,
            down_args,
            ct_down_addr_base);
    };

    // The leader gathers into the slot it has reserved for the current block. Block 0's slot is
    // claimed here; every later block's is claimed by the previous block's broadcast step, which is
    // what lets the other cores scatter into it (see leader_gather_broadcast_block).
    CircularBuffer cb_act(cb_act_id);
    if constexpr (IsLeader) {
        const uint32_t first_block = num_active < experts_block ? num_active : experts_block;
        cb_act.reserve_back(first_block * i_tiles);
    }

    uint32_t gather_target = 0;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t first_expert = blk * experts_block;
        const uint32_t remaining = num_active - first_expert;
        const uint32_t block_experts = remaining < experts_block ? remaining : experts_block;
        const uint32_t block_act_tiles = block_experts * i_tiles;
        const uint32_t next_block_experts =
            blk + 1u < num_blocks
                ? (remaining - block_experts < experts_block ? remaining - block_experts : experts_block)
                : 0u;

        // ---- Phase 1: gate_up weights for the block's experts (SwiGLU cores only). ----
        // cb_weights is double-buffered, so the reader can hold one expert's slice ready while
        // compute consumes the previous expert's -- overlapping data movement with computation.
        {
            DeviceZoneScopedN("FE_PHASE1_GATE_UP");
            if (swiglu_core) {
                for (uint32_t j = 0; j < block_experts; ++j) {
                    fetch_weight_one(
                        noc,
                        cb_bcast_id,
                        cb_weights_id,
                        first_expert + j,
                        gate_up_slice_tiles,
                        gate_up_reserve_tiles,
                        gate_up_tile_bytes,
                        shard_id,
                        gate_up_args,
                        ct_gu_addr_base);
                }
            }
        }

        // Build this core's routing-weight scalar tiles for the block's experts, for the down-output
        // weighted accumulation. Deliberately placed after phase 1: it is a few thousand serial L1
        // stores that nothing needs until the down phase, so running it here buries it in the
        // gather/broadcast wait below instead of delaying the first DRAM read.
        {
            DeviceZoneScopedN("FE_RSCALARS");
            build_routing_scalars(cb_bcast_id, cb_rscalar_id, first_expert, block_experts, weight_base, batch);
        }

        // ---- Down-weight prefetch across the sync (receivers only). ----
        // The down weights depend on nothing but the expert ids, while the gather/broadcast below is
        // a chip-wide barrier during which DRAM would otherwise sit idle. Receivers therefore start
        // streaming down slices before waiting on the broadcast. `down_prefetch` is capped below the
        // CB's slot count by the host so that reserve_back cannot block forever here -- the reader
        // must reach receiver_recv_act_block to publish cb_act to its compute kernel. It is also
        // capped at the block, whose experts are the only ones compute will consume next.
        // The leader is excluded: its broadcast gates every other core, so it syncs first.
        uint32_t j_down = 0;
        if constexpr (!IsLeader) {
            DeviceZoneScopedN("FE_DOWN_PREFETCH");
            const uint32_t prefetch = down_prefetch < block_experts ? down_prefetch : block_experts;
            for (; j_down < prefetch; ++j_down) {
                fetch_down(first_expert + j_down);
            }
        }

        // ---- Synchronization: gather the block's activations + broadcast the block. ----
        {
            DeviceZoneScopedN("FE_SYNC");
            if constexpr (IsLeader) {
                // Cumulative target: the semaphore counts up across blocks so it never needs a reset.
                gather_target += num_producers * block_experts + slot_free_acks_per_block;
                const uint32_t act_l1 = cb_act.get_write_ptr();
                leader_gather_broadcast_block(
                    noc,
                    cb_act_id,
                    act_l1,
                    block_act_tiles,
                    next_block_experts * i_tiles,
                    act_tile_bytes,
                    gather_target,
                    sem_gather_id,
                    sem_bcast_id,
                    /*blocks_done=*/blk + 1u,
                    mcast_start_x,
                    mcast_start_y,
                    mcast_end_x,
                    mcast_end_y,
                    num_dests);
            } else {
                receiver_recv_act_block(
                    noc,
                    cb_act_id,
                    block_act_tiles,
                    sem_bcast_id,
                    /*blocks_done=*/blk + 1u,
                    /*send_slot_free_ack=*/!swiglu_core,
                    sem_gather_id,
                    leader_noc_x,
                    leader_noc_y);
            }
        }

        // ---- Phase 2: the block's down weights not already prefetched. ----
        // cb_down_w reuses the (now-dead) cb_weights CB, which is large enough for several down
        // slices, so the same reader/compute overlap applies.
        {
            DeviceZoneScopedN("FE_PHASE2_DOWN");
            for (; j_down < block_experts; ++j_down) {
                fetch_down(first_expert + j_down);
            }
        }
    }
}
