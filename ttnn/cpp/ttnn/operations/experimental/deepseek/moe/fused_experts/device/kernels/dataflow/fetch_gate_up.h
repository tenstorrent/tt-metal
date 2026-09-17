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
//   2. GATHER: each SwiGLU core's writer copies its act tiles into the cb_act slot of the hub
//      that owns them (a single NoC write) and bumps BOTH hubs' gather semaphores. The I dim is
//      split between the two hubs at `split_col` tiles per expert, so hub0 receives the low
//      columns of every expert of the block and hub1 the high ones. After all chunks land, each
//      hub's slot holds its half of the block's act[B, I] per expert (i_tiles == I/32 tiles each,
//      in K order) and, because each hub receives the other's half too (step 3), the whole block.
//   3. BROADCAST: the two hubs multicast their own half of every expert to every other core's
//      cb_act (same L1 address), each on a DIFFERENT NoC, and each publishes one increment of the
//      broadcast semaphore. Once a core has seen num_hubs increments it has both halves, so it
//      publishes the slot to its compute. One multicast sender per NoC is required: two senders
//      into the same rectangle on one NoC circular-wait on overlapping path reservations.
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

// Which role a core plays in one block's activation gather + broadcast.
//   0 PLAIN: receives the block; never multicasts.
//   1 HUB0 : gathers its half of the I dim, multicasts it, and publishes the block once BOTH hubs
//            have.
//   2 HUB1 : the same, on the other NoC and for the other half.
//
// The two hubs are the two opposite corners of the multicast rectangle. The SwiGLU I dim is split
// at `split_col` act tiles per expert: hub0 owns tiles [0, split_col) of every expert of the block,
// hub1 owns [split_col, i_tiles). Each producer scatters its slice to the hub that owns it (the
// slice's first column decides), so each hub gathers and multicasts half the block.
//
// Splitting the gather AND the broadcast in half is the point -- but the multicast must be issued
// on two DIFFERENT NoCs: two multicast senders whose paths into the same destination rectangle
// overlap circular-wait, so exactly one multicast sender per NoC is allowed. That is the same
// constraint matmul_decode's two-hub gather works around by splitting its two senders across the
// NOCs (hub0 on NOC0, hub1 on NOC1). Here hub1's reader is compiled onto NOC1 for that reason.
//
// `split_col == i_tiles` with `num_hubs == 1` degenerates to the original single-hub pipeline:
// hub0 owns the whole I dim and hub1 is never used. Every core waits for `num_hubs` completion
// increments per block, so both shapes share one code path.
struct ActGatherConfig {
    uint32_t role = 0;
    uint32_t hub0_x = 0;
    uint32_t hub0_y = 0;
    uint32_t hub1_x = 0;
    uint32_t hub1_y = 0;
    uint32_t split_col = 0;  // first act tile of each expert that hub1 owns (== i_tiles if one hub)
    uint32_t num_hubs = 1;   // 1 or 2
};

// NoC 1 traverses a multicast rectangle from high to low coordinates, so its start/end corners are
// swapped relative to NoC 0's. The host always hands over the ascending (bbox min -> bbox max)
// corners, so a kernel running on NoC 1 swaps them here. Unicast coordinates are never inverted.
FORCE_INLINE void orient_mcast(uint32_t& x_start, uint32_t& y_start, uint32_t& x_end, uint32_t& y_end) {
    if (noc_index == 1) {
        const uint32_t tx = x_start;
        const uint32_t ty = y_start;
        x_start = x_end;
        y_start = y_end;
        x_end = tx;
        y_end = ty;
    }
}

// Hub side of ONE BLOCK's activation gather + broadcast.
//
// By the time this runs, every SwiGLU core has produced the block's activation slices (phase 1) and
// its writer has scattered them into the owning hub's cb_act slot for this block -- the block's
// expert j chunk for core idx at tile offset (j*i_tiles + idx*swiglu_tiles). The slot therefore
// holds the block's [n, B, I] activations once `gather_target` bumps have landed (num_producers per
// expert, counted cumulatively across blocks, plus one ack per block from any core that has no
// chunk to send). Each bump is delivered to BOTH hubs, so each hub sees the full count and can
// trust that EVERY core is ready for the slot it is about to overwrite -- not just the cores that
// scatter into it. Within a block cb_act is never reused, so no per-expert back-pressure is needed.
//
// Both hubs run this concurrently, each multicasting only its own half of every expert, on its own
// NoC.
//
// `next_block_act_tiles` is the next block's size, or 0 if this is the last block (or the only one).
// Reserving that space *before* the broadcast is what makes blocking safe: it returns only once this
// core's compute has released the slot the next block will reuse, so a core that has seen this
// broadcast may scatter into the hub's next slot without any further handshake. Note the reserve
// charges `block_act_tiles + next_block_act_tiles`: the current block's data is already written but
// its push_back is deferred to the end (it must not be published to the local compute until the
// peer hub's half has landed), so its pages have to be charged here as well. That reproduces
// exactly the free-space wait the single-hub pipeline got from pushing first.
//
// `send_slot_free_ack` must be set for a hub core that owns NO I slice (hub1 at a small TP I): like
// any other core it owes the OTHER hub one ack per block, and because it runs this path rather than
// receiver_recv_act_block nothing else would send it -- the peer hub's gather target would then
// never be reached. It must NOT ack itself: its own slot was already proven free by its own reserve
// one block earlier, and because that self-bump would arrive early it could stand in for an ack
// still owed by some other core and release this hub before that core was ready (hence the
// -1 applied to this hub's own target increment in run_reader_loop).
inline void hub_gather_broadcast_block(
    const Noc& noc,
    uint32_t cb_act_id,
    uint32_t act_l1,
    uint32_t block_act_tiles,
    uint32_t next_block_act_tiles,
    uint32_t act_tile_bytes,
    uint32_t block_experts,
    uint32_t i_tiles,
    uint32_t region_first,
    uint32_t region_cols,
    uint32_t gather_target,
    uint32_t sem_gather_id,
    uint32_t sem_bcast_id,
    uint32_t blocks_done,
    uint32_t num_dests,
    const ActGatherConfig& gather,
    bool send_slot_free_ack,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y) {
    CircularBuffer cb_act(cb_act_id);
    const bool is_hub0 = gather.role == 1;
    const uint32_t hub_self_x = is_hub0 ? gather.hub0_x : gather.hub1_x;
    const uint32_t hub_self_y = is_hub0 ? gather.hub0_y : gather.hub1_y;

    // This block's synchronization point: wait for each of its experts' chunks from every SwiGLU
    // core (and one ack per block from every core that has no chunk).
    Semaphore<>(sem_gather_id).wait_min(gather_target);

    // Claim space for the current block (written, not yet published) plus the next one before
    // announcing this block (see above).
    if (next_block_act_tiles > 0) {
        cb_act.reserve_back(block_act_tiles + next_block_act_tiles);
    }

    // A hub with no I slice of its own owes the peer hub the same per-block ack a non-producer
    // receiver sends; this reserve is the "my compute is past the previous block" evidence behind it.
    if (send_slot_free_ack) {
        Semaphore<>(sem_gather_id)
            .up(noc, is_hub0 ? gather.hub1_x : gather.hub0_x, is_hub0 ? gather.hub1_y : gather.hub0_y, 1);
    }

    // Multicast this hub's region of every expert of the block, to every other core's cb_act (same
    // L1 address everywhere). When the hub owns the whole I dim its region IS the whole block, so
    // one multicast covers it (this is the single-hub case, unchanged); otherwise the region is a
    // per-expert column range and neither hub's region is contiguous over the block, so it is one
    // multicast per expert. The two hubs' regions are disjoint and they run on different NoCs, so
    // no multicast path is shared.
    orient_mcast(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y);
    if (region_first == 0u && region_cols == i_tiles) {
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
    } else if (region_cols > 0) {
        const uint32_t region_bytes = region_cols * act_tile_bytes;
        for (uint32_t j = 0; j < block_experts; ++j) {
            const uint32_t region_off = (j * i_tiles + region_first) * act_tile_bytes;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(act_l1 + region_off),
                MulticastEndpoint{},
                region_bytes,
                num_dests,
                {.offset_bytes = 0},
                {.noc_x_start = mcast_start_x,
                 .noc_y_start = mcast_start_y,
                 .noc_x_end = mcast_end_x,
                 .noc_y_end = mcast_end_y,
                 .addr = act_l1 + region_off},
                /*linked=*/false);
        }
        noc.async_write_barrier();
    }

    // Publish this hub's half: one increment on every OTHER core of the rectangle, plus an atomic
    // increment of our own (a plain local up() can race with the peer hub's multicast increment; a
    // NoC atomic cannot). The semaphore counts hub broadcasts cumulatively rather than carrying a
    // block number, so it never has to be reset -- a reset would race with a core still reading the
    // previous value -- and a core simply waits for num_hubs increments per block.
    Semaphore<> sem(sem_bcast_id);
    sem.inc_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 1, num_dests);
    sem.up(noc, hub_self_x, hub_self_y, 1);
    noc.async_atomic_barrier();

    // The slot is complete only once BOTH hubs have multicast it: the peer hub's half has to have
    // landed in this core's L1 (and in every other core's) before the local compute may read it.
    sem.wait_min(blocks_done);
    cb_act.push_back(block_act_tiles);
}

// Non-hub side: wait for the hubs' broadcast of one block of activations, then publish it (already
// resident, written straight into this core's cb_act by the multicasts) to the local compute.
//
// The space is claimed first, so the block is only handed to the hubs once compute has released
// whatever block previously occupied it. For a core that scatters, that release is already implied
// by the scatter itself -- it cannot happen until its compute has moved past the previous block --
// but a core with no slice of the I dim sends no chunk, so it acks the gather here instead. The ack
// goes to BOTH hubs: each hub needs evidence that every core is done with the slot it is about to
// overwrite, including the cores that scatter into its peer. Without that a hub would have no
// evidence about the cores whose chunks it does not gather.
inline void receiver_recv_act_block(
    const Noc& noc,
    uint32_t cb_act_id,
    uint32_t block_act_tiles,
    uint32_t sem_bcast_id,
    uint32_t sem_gather_id,
    uint32_t blocks_done,
    bool send_slot_free_ack,
    const ActGatherConfig& gather) {
    CircularBuffer cb_act(cb_act_id);
    cb_act.reserve_back(block_act_tiles);
    if (send_slot_free_ack) {
        Semaphore<> sem_gather(sem_gather_id);
        sem_gather.up(noc, gather.hub0_x, gather.hub0_y, 1);
        if (gather.num_hubs > 1) {
            sem_gather.up(noc, gather.hub1_x, gather.hub1_y, 1);
        }
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
// TILE LAYOUT (bf16, width 32, height `tile_h` in {1,2,4,8,16,32}):
//   * face_r_dim = min(tile_h, 16); num_face_rows_per_tile = ceil(tile_h/16) (1 for tile_h<=16, 2
//     for tile_h==32); num_faces = num_face_rows_per_tile * 2 (two 16-col faces side by side).
//   * Faces are stored in raster order: (0,1) is the top pair of 16x16 (or face_r_dim x 16) faces,
//     (2,3) the bottom pair (only present when tile_h > 16). Row r of the tile lands at face row
//     `r % face_r_dim` of the face pair (r / face_r_dim) * 2, at column groups {0..15, 16..31}.
//   * Every store is a 32-bit write of two bf16 lanes (halved store count on the critical path).
//     Writing every real row (rather than zeroing the tile first) touches each word exactly once;
//     rows past `batch` don't need to be visited at all for tile_h == batch, and for tile_h > batch
//     the padding-row region is pre-zeroed by the reserve_back allocation.
//
// `tile_bytes` is the actual bytes-per-tile derived by the host from `tile.get_tile_size(bf16)`;
// it is num_faces * face_r_dim * 16 * 2 -- the exact stride between consecutive expert tiles in
// cb_rscalar. `face_r_dim` and `num_face_rows` come from the same host tile descriptor.
inline void build_routing_scalars(
    uint32_t cb_bcast_id,
    uint32_t cb_rscalar_id,
    uint32_t first_expert,
    uint32_t count,
    uint32_t weight_base,
    uint32_t batch,
    uint32_t tile_h,
    uint32_t face_r_dim,
    uint32_t num_face_rows,
    uint32_t tile_bytes) {
    CircularBuffer cb_bcast(cb_bcast_id);
    CoreLocalMem<volatile uint32_t> bcast(cb_bcast.get_write_ptr());

    CircularBuffer cb_rscalar(cb_rscalar_id);
    cb_rscalar.reserve_back(count);
    const uint32_t rscalar_l1 = cb_rscalar.get_write_ptr();

    // 32-bit words per face and per face-row (bf16, width 32 => 16-wide faces => 8 words per row).
    const uint32_t face_words = (face_r_dim * 16u * 2u) / 4u;
    constexpr uint32_t kRowWords = 16 / 2;
    // When tile_h < 32 (and specifically tile_h <= face_r_dim), the bottom half of the tile does
    // not exist as storage. `num_face_rows == 1` skips the second face pair entirely.
    const uint32_t real_rows = batch < tile_h ? batch : tile_h;
    (void)num_face_rows;  // implicit in face_r_dim * num_face_rows == tile_h; kept for clarity.
    for (uint32_t e = 0; e < count; ++e) {
        CoreLocalMem<volatile uint32_t> tile(rscalar_l1 + e * tile_bytes);
        // Pre-zero: for tile_h > batch we need padding rows to be 0, and the CB is not guaranteed
        // clean between blocks. Cheaper than a per-row branch inside the splat loop, and the whole
        // tile fits in <= 2 KB.
        if (real_rows < tile_h) {
            const uint32_t words = tile_bytes / 4u;
            for (uint32_t w = 0; w < words; ++w) {
                tile[w] = 0;
            }
        }
        for (uint32_t r = 0; r < real_rows; ++r) {
            const uint32_t w_bf16 = bcast[weight_base + (first_expert + e) * batch + r] >> 16;
            const uint32_t w_pair = (w_bf16 << 16) | w_bf16;
            const uint32_t face_pair = (r / face_r_dim) * 2u * face_words;
            const uint32_t row_off = (r % face_r_dim) * kRowWords;
            for (uint32_t j = 0; j < kRowWords; ++j) {
                tile[face_pair + row_off + j] = w_pair;
                tile[face_pair + face_words + row_off + j] = w_pair;
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
//            hub that owns its I slice.
//   Sync:    the two hubs gather the block's activations (one half of the I dim each) and each
//            multicasts its half to every core; everyone waits for both halves -- and, while
//            waiting, prefetches up to `down_prefetch` of the block's down slices so DRAM keeps
//            streaming through the barrier.
//   Phase 2: fetch the block's remaining down slices. The compute kernel runs the down matmul for
//            the block's experts against the now-resident activations and accumulates.
//
// When `num_expert_groups > 1` (the 6-expert / 96-core path), the 12x8 grid is partitioned into
// groups of `cores_per_expert` cores, one group per hit expert. Each core covers `i_shards_per_core`
// gate_up/I shards (I/32 / 16; one shard at TP I=512, four at I=2048) and `shards_per_core` down/H
// shards (always 4 of the 64). The gather/broadcast is per-group (two hubs per group, the group
// rectangle's two corners), and the 6 groups' matching H-slices are reduced onto group 0 (see the
// writer / compute kernels).
template <typename GateUpArgs, typename DownArgs>
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
    // Routing-scalar tile geometry: bf16 tile of the input's height (matches the token-row-shaped
    // pipeline), width fixed at 32. Passed as regular args so callers can plumb them from their
    // compile-time args without changing this template's signature per tile shape.
    uint32_t rscalar_tile_h,
    uint32_t rscalar_face_r_dim,
    uint32_t rscalar_num_face_rows,
    uint32_t rscalar_tile_bytes,
    uint32_t cores_per_expert,
    uint32_t shards_per_core,
    uint32_t i_shards_per_core,
    uint32_t num_expert_groups,
    const ActGatherConfig& gather,
    uint32_t sem_reduce_id,
    uint32_t cb_reduce_id) {
    constexpr uint32_t kOutTilesPerCore = 2;
    const bool is_hub = gather.role != 0;

    // ---- 6-expert path: 16 cores per expert; I-shards scale with local I, H-shards stay 4. ----
    if (num_expert_groups > 1) {
        const uint32_t local_idx = core_index % cores_per_expert;
        const uint32_t expert_group = core_index / cores_per_expert;
        const uint32_t swiglu_tiles = i_tiles / (cores_per_expert * i_shards_per_core);
        const uint32_t gate_up_slice_tiles = k_tiles * (2u * swiglu_tiles);
        const bool is_root = expert_group == 0;
        const uint32_t out_tiles = shards_per_core * kOutTilesPerCore;
        const uint32_t reduce_tiles = (num_expert_groups - 1u) * out_tiles;

        CircularBuffer cb_act(cb_act_id);
        if (is_hub) {
            cb_act.reserve_back(i_tiles);
        }
        if (is_root) {
            CircularBuffer cb_reduce(cb_reduce_id);
            cb_reduce.reserve_back(reduce_tiles);
        }

        {
            DeviceZoneScopedN("FE_PHASE1_GATE_UP");
            for (uint32_t s = 0; s < i_shards_per_core; ++s) {
                fetch_weight_one(
                    noc,
                    cb_bcast_id,
                    cb_weights_id,
                    expert_group,
                    gate_up_slice_tiles,
                    gate_up_reserve_tiles,
                    gate_up_tile_bytes,
                    local_idx * i_shards_per_core + s,
                    gate_up_args,
                    ct_gu_addr_base);
            }
        }

        {
            DeviceZoneScopedN("FE_RSCALARS");
            build_routing_scalars(
                cb_bcast_id,
                cb_rscalar_id,
                expert_group,
                1u,
                weight_base,
                batch,
                rscalar_tile_h,
                rscalar_face_r_dim,
                rscalar_num_face_rows,
                rscalar_tile_bytes);
        }

        uint32_t j_down = 0;
        if (!is_hub) {
            DeviceZoneScopedN("FE_DOWN_PREFETCH");
            const uint32_t prefetch = down_prefetch < shards_per_core ? down_prefetch : shards_per_core;
            for (; j_down < prefetch; ++j_down) {
                fetch_weight_one(
                    noc,
                    cb_bcast_id,
                    cb_down_w_id,
                    expert_group,
                    down_slice_tiles,
                    down_reserve_tiles,
                    down_tile_bytes,
                    local_idx * shards_per_core + j_down,
                    down_args,
                    ct_down_addr_base);
            }
        }

        {
            DeviceZoneScopedN("FE_SYNC");
            if (is_hub) {
                // Single expert per group, so each hub multicasts its own column range of that one
                // expert; `num_producers` chunks is the whole gather target (every core of the group
                // owns a chunk, so there are no slot-free acks).
                const bool is_hub0 = gather.role == 1;
                const uint32_t region_first = is_hub0 ? 0u : gather.split_col;
                const uint32_t region_cols = is_hub0 ? gather.split_col : (i_tiles - gather.split_col);
                const uint32_t act_l1 = cb_act.get_write_ptr();
                hub_gather_broadcast_block(
                    noc,
                    cb_act_id,
                    act_l1,
                    i_tiles,
                    /*next_block_act_tiles=*/0u,
                    act_tile_bytes,
                    /*block_experts=*/1u,
                    i_tiles,
                    region_first,
                    region_cols,
                    /*gather_target=*/num_producers,
                    sem_gather_id,
                    sem_bcast_id,
                    /*blocks_done=*/gather.num_hubs,
                    num_dests,
                    gather,
                    /*send_slot_free_ack=*/false,
                    mcast_start_x,
                    mcast_start_y,
                    mcast_end_x,
                    mcast_end_y);
            } else {
                receiver_recv_act_block(
                    noc,
                    cb_act_id,
                    i_tiles,
                    sem_bcast_id,
                    sem_gather_id,
                    /*blocks_done=*/gather.num_hubs,
                    /*send_slot_free_ack=*/false,
                    gather);
            }
        }

        {
            DeviceZoneScopedN("FE_PHASE2_DOWN");
            for (; j_down < shards_per_core; ++j_down) {
                fetch_weight_one(
                    noc,
                    cb_bcast_id,
                    cb_down_w_id,
                    expert_group,
                    down_slice_tiles,
                    down_reserve_tiles,
                    down_tile_bytes,
                    local_idx * shards_per_core + j_down,
                    down_args,
                    ct_down_addr_base);
            }
        }

        if (is_root) {
            Semaphore<>(sem_reduce_id).wait(num_expert_groups - 1u);
            CircularBuffer cb_reduce(cb_reduce_id);
            cb_reduce.push_back(reduce_tiles);
        }
        return;
    }

    // Both the gate_up and the down DRAM shards for this core are shard `core_index`; the
    // first `num_producers` cores also own a slice of the SwiGLU I dim.
    (void)i_shards_per_core;
    const bool swiglu_core = core_index < num_producers;
    const uint32_t shard_id = core_index;
    const uint32_t swiglu_tiles = i_tiles / num_producers;
    const uint32_t gate_up_slice_tiles = k_tiles * (2u * swiglu_tiles);

    const uint32_t num_blocks = (num_active + experts_block - 1u) / experts_block;

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

    // The hubs gather into the slot they have reserved for the current block. Block 0's slot is
    // claimed here; every later block's is claimed by the previous block's broadcast step, which is
    // what lets the other cores scatter into it (see hub_gather_broadcast_block).
    CircularBuffer cb_act(cb_act_id);
    if (is_hub) {
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
            build_routing_scalars(
                cb_bcast_id,
                cb_rscalar_id,
                first_expert,
                block_experts,
                weight_base,
                batch,
                rscalar_tile_h,
                rscalar_face_r_dim,
                rscalar_num_face_rows,
                rscalar_tile_bytes);
        }

        // ---- Down-weight prefetch across the sync (receivers only). ----
        // The down weights depend on nothing but the expert ids, while the gather/broadcast below is
        // a chip-wide barrier during which DRAM would otherwise sit idle. Receivers therefore start
        // streaming down slices before waiting on the broadcast. `down_prefetch` is capped below the
        // CB's slot count by the host so that reserve_back cannot block forever here -- the reader
        // must reach receiver_recv_act_block to publish cb_act to its compute kernel. It is also
        // capped at the block, whose experts are the only ones compute will consume next.
        // The hub is excluded from the prefetch: its broadcast gates every other core, so it syncs
        // first.
        uint32_t j_down = 0;
        if (!is_hub) {
            DeviceZoneScopedN("FE_DOWN_PREFETCH");
            const uint32_t prefetch = down_prefetch < block_experts ? down_prefetch : block_experts;
            for (; j_down < prefetch; ++j_down) {
                fetch_down(first_expert + j_down);
            }
        }

        // ---- Synchronization: gather the block's activations + broadcast the block. ----
        {
            DeviceZoneScopedN("FE_SYNC");
            if (is_hub) {
                // Cumulative target: the semaphore counts up across blocks so it never needs a reset.
                // Per block this hub receives one scatter bump per producer per expert, plus one ack
                // from each core that owns no chunk -- except its own, which it does not send itself
                // (see hub_gather_broadcast_block): hence the -1 when this hub owns no I slice.
                gather_target +=
                    num_producers * block_experts + (num_dests + 1u - num_producers) - (swiglu_core ? 0u : 1u);
                const bool is_hub0 = gather.role == 1;
                // Hub0 owns I-tiles [0, split_col) of every expert of the block, hub1 the rest.
                const uint32_t region_first = is_hub0 ? 0u : gather.split_col;
                const uint32_t region_cols = is_hub0 ? gather.split_col : (i_tiles - gather.split_col);
                const uint32_t act_l1 = cb_act.get_write_ptr();
                hub_gather_broadcast_block(
                    noc,
                    cb_act_id,
                    act_l1,
                    block_act_tiles,
                    next_block_experts * i_tiles,
                    act_tile_bytes,
                    block_experts,
                    i_tiles,
                    region_first,
                    region_cols,
                    gather_target,
                    sem_gather_id,
                    sem_bcast_id,
                    /*blocks_done=*/(blk + 1u) * gather.num_hubs,
                    num_dests,
                    gather,
                    /*send_slot_free_ack=*/!swiglu_core,
                    mcast_start_x,
                    mcast_start_y,
                    mcast_end_x,
                    mcast_end_y);
            } else {
                receiver_recv_act_block(
                    noc,
                    cb_act_id,
                    block_act_tiles,
                    sem_bcast_id,
                    sem_gather_id,
                    /*blocks_done=*/(blk + 1u) * gather.num_hubs,
                    /*send_slot_free_ack=*/!swiglu_core,
                    gather);
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
