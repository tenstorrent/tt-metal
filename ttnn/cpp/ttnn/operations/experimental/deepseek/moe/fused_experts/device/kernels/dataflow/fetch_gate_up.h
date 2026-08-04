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
// PER-EXPERT PIPELINE (run by the reader on every core, in lock-step across the chip):
//   1. gate_up matmul + SwiGLU produces, on each of the I/64 SwiGLU cores, a 2-tile
//      (64-column) slice of the activation act[1, I] (its I-columns [idx*64, idx*64+64)).
//   2. GATHER: each SwiGLU core's writer copies its 2 act tiles into core {0,0}'s cb_act
//      at tile offset idx*2 (a single NoC write to the leader) and bumps the leader's
//      gather semaphore. After all chunks land, {0,0}'s cb_act holds the full act[1, I]
//      (i_tiles == I/32 tiles, in K order).
//   3. BROADCAST: {0,0} multicasts its full cb_act to every other core's cb_act (same L1
//      address) and sets the broadcast semaphore. Now every core has the full activation.
//   4. DOWN matmul: each of the 64 cores multiplies the full act[1, I] by its own down
//      weight shard ([I, H/64] -> down_slice_tiles tiles) to produce its 2-tile (64-column)
//      slice of the output row[1, H]; the compute kernel scales it by the expert's routing
//      weight and accumulates across experts into the single [1, 1, H] DRAM output.
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
// weight CB. The CB's reserve_back/push_back provide compute-side flow control: with two
// slots the reader can hold one expert's data ready while compute consumes the previous
// expert's, decoupling the reader from compute (see tech_reports/Saturating_DRAM_bandwidth --
// "double buffered, to overlap the data movement with computation").
//
// `cb_weights_id` must be a CB with total_size >= 2 * slice_tiles * page_size.
// `ct_w_addr_base` is the compile-time-args offset of the per-expert weight base addresses.
template <typename WeightArgs>
inline void fetch_weight_one(
    const Noc& noc,
    uint32_t cb_bcast_id,
    uint32_t cb_weights_id,
    uint32_t i,
    uint32_t slice_tiles,
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
    cb_weights.reserve_back(slice_tiles);
    ShardView w_shard(w);
    noc.async_read(w_shard, cb_weights, slice_bytes, {.shard_id = shard_id}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_weights.push_back(slice_tiles);
}

// Leader ({0,0}) side of the SINGLE activation gather + broadcast for ALL experts.
//
// By the time this runs, every SwiGLU core has produced all `num_active` activation slices
// (phase 1) and its writer has scattered them into the leader's cb_act -- expert e's chunk
// for core idx at tile offset (e*i_tiles + idx*2). cb_act therefore holds the full
// [num_active, I] activation block once all `num_producers * num_active` chunks have landed.
// The leader then multicasts the whole block to every other core in one shot and publishes
// it locally. Because cb_act is never reused across experts (it holds them all at once), no
// per-expert back-pressure is needed -- a single gather wait + a single broadcast suffice.
inline void leader_gather_broadcast_all(
    const Noc& noc,
    uint32_t cb_act_id,
    uint32_t act_total_tiles,
    uint32_t act_tile_bytes,
    uint32_t gather_count,
    uint32_t sem_gather_id,
    uint32_t sem_bcast_id,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y,
    uint32_t num_dests) {
    CircularBuffer cb_act(cb_act_id);
    cb_act.reserve_back(act_total_tiles);
    const uint32_t act_l1 = cb_act.get_write_ptr();

    // Single synchronization point: wait for every expert's chunk from every SwiGLU core.
    Semaphore<>(sem_gather_id).wait_min(gather_count);

    // Broadcast the whole [num_active, I] activation block to every other core's cb_act.
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(act_l1),
        MulticastEndpoint{},
        act_total_tiles * act_tile_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = act_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    Semaphore<> sem(sem_bcast_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);

    cb_act.push_back(act_total_tiles);
}

// Non-leader side: wait for the leader's single broadcast of the full [num_active, I]
// activation block, then publish it (already resident) to the local compute kernel.
inline void receiver_recv_act_all(uint32_t cb_act_id, uint32_t act_total_tiles, uint32_t sem_bcast_id) {
    Semaphore<>(sem_bcast_id).wait_min(1);
    CircularBuffer cb_act(cb_act_id);
    cb_act.reserve_back(act_total_tiles);
    cb_act.push_back(act_total_tiles);
}

// Build one bf16 SCALAR-broadcast tile per active expert from the routing-weight scalars the
// leader appended to cb_bcast (at index weight_base + e, as fp32 bit patterns). The down-output
// multiply reads element [0,0] of the broadcast tile; following the canonical reduce/bcast
// scaler layout, the value is splatted into the first row of all four 16x16 faces and the rest
// zeroed. Runs once per core (the bcast buffer is resident before the reader loop starts).
inline void build_routing_scalars(
    uint32_t cb_bcast_id, uint32_t cb_rscalar_id, uint32_t num_active, uint32_t weight_base) {
    CircularBuffer cb_bcast(cb_bcast_id);
    CoreLocalMem<volatile uint32_t> bcast(cb_bcast.get_write_ptr());

    CircularBuffer cb_rscalar(cb_rscalar_id);
    cb_rscalar.reserve_back(num_active);
    const uint32_t rscalar_l1 = cb_rscalar.get_write_ptr();

    constexpr uint32_t kTileElems = 1024;            // 32x32 bf16 elements per tile
    constexpr uint32_t kTileBytes = kTileElems * 2;  // bf16 tile = 2048 bytes
    // Zero and splat through 32-bit stores (two bf16 lanes each): this loop runs on every core
    // ahead of the DRAM-bound phases, so halving its store count is pure critical-path savings.
    constexpr uint32_t kTileWords = kTileElems / 2;
    for (uint32_t e = 0; e < num_active; ++e) {
        CoreLocalMem<volatile uint32_t> tile(rscalar_l1 + e * kTileBytes);
        for (uint32_t j = 0; j < kTileWords; ++j) {
            tile[j] = 0;
        }
        // fp32 bit pattern -> bf16 == high 16 bits, packed into both lanes of a 32-bit store.
        const uint32_t w_bf16 = bcast[weight_base + e] >> 16;
        const uint32_t w_pair = (w_bf16 << 16) | w_bf16;
        for (uint32_t k = 0; k < 4; ++k) {      // 4 faces
            for (uint32_t j = 0; j < 8; ++j) {  // first row (16 bf16) of each face
                tile[k * 128 + j] = w_pair;
            }
        }
    }
    cb_rscalar.push_back(num_active);
}

// Per-core reader loop shared by all DM reader kernels, structured in two phases around a
// single synchronization:
//   Phase 1: fetch this core's gate_up slice for ALL experts (SwiGLU cores only). The compute
//            kernel produces every expert's SwiGLU activation, which the writer scatters to
//            the leader.
//   Sync:    the leader gathers all experts' activations and broadcasts the whole
//            [num_active, I] block to every core in one shot; everyone else waits for it --
//            and, while waiting, prefetches up to `down_prefetch` down slices so DRAM keeps
//            streaming through the barrier.
//   Phase 2: fetch this core's remaining down slices. The compute kernel runs the down matmul
//            for every expert against the now-resident activations.
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
    uint32_t down_prefetch) {
    // Both the gate_up and the down DRAM shards for this core are shard `core_index`; the
    // first `num_producers` cores also own a slice of the SwiGLU I dim.
    const bool swiglu_core = core_index < num_producers;
    const uint32_t shard_id = core_index;
    const uint32_t swiglu_tiles = i_tiles / num_producers;

    // ---- Phase 1: gate_up weights for all experts (SwiGLU cores only). ----
    // cb_weights is double-buffered, so the reader can hold one expert's slice ready while
    // compute consumes the previous expert's -- overlapping data movement with computation.
    {
        DeviceZoneScopedN("FE_PHASE1_GATE_UP");
        if (swiglu_core) {
            const uint32_t gate_up_slice_tiles = k_tiles * (2u * swiglu_tiles);
            for (uint32_t e = 0; e < num_active; ++e) {
                fetch_weight_one(
                    noc,
                    cb_bcast_id,
                    cb_weights_id,
                    e,
                    gate_up_slice_tiles,
                    gate_up_tile_bytes,
                    shard_id,
                    gate_up_args,
                    ct_gu_addr_base);
            }
        }
    }

    // Build this core's routing-weight scalar tiles (one per active expert) for the down-output
    // weighted accumulation. Deliberately placed after phase 1: it is a few thousand serial L1
    // stores that nothing needs until the down phase, so running it here buries it in the
    // gather/broadcast wait below instead of delaying the first DRAM read.
    {
        DeviceZoneScopedN("FE_RSCALARS");
        build_routing_scalars(cb_bcast_id, cb_rscalar_id, num_active, weight_base);
    }

    auto fetch_down = [&](uint32_t e) {
        fetch_weight_one(
            noc,
            cb_bcast_id,
            cb_down_w_id,
            e,
            down_slice_tiles,
            down_tile_bytes,
            shard_id,
            down_args,
            ct_down_addr_base);
    };

    // ---- Down-weight prefetch across the sync (receivers only). ----
    // The down weights depend on nothing but the expert ids, while the gather/broadcast below is
    // a chip-wide barrier during which DRAM would otherwise sit idle. Receivers therefore start
    // streaming down slices before waiting on the broadcast. `down_prefetch` is capped below the
    // CB's slot count by the host so that reserve_back cannot block forever here -- the reader
    // must reach receiver_recv_act_all to publish cb_act to its compute kernel.
    // The leader is excluded: its broadcast gates every other core, so it syncs first.
    uint32_t e_down = 0;
    if constexpr (!IsLeader) {
        DeviceZoneScopedN("FE_DOWN_PREFETCH");
        for (; e_down < down_prefetch; ++e_down) {
            fetch_down(e_down);
        }
    }

    // ---- Single synchronization: gather all activations + broadcast the whole block. ----
    const uint32_t act_total_tiles = num_active * i_tiles;
    {
        DeviceZoneScopedN("FE_SYNC");
        if constexpr (IsLeader) {
            leader_gather_broadcast_all(
                noc,
                cb_act_id,
                act_total_tiles,
                act_tile_bytes,
                num_producers * num_active,
                sem_gather_id,
                sem_bcast_id,
                mcast_start_x,
                mcast_start_y,
                mcast_end_x,
                mcast_end_y,
                num_dests);
        } else {
            receiver_recv_act_all(cb_act_id, act_total_tiles, sem_bcast_id);
        }
    }

    // ---- Phase 2: the down weights not already prefetched. ----
    // cb_down_w reuses the (now-dead) cb_weights CB, which is large enough for several down
    // slices, so the same reader/compute overlap applies.
    {
        DeviceZoneScopedN("FE_PHASE2_DOWN");
        for (; e_down < num_active; ++e_down) {
            fetch_down(e_down);
        }
    }
}
