// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer kernel (reader RISC, NOC_0). Stages the tokens one ring direction's senders are about to want
// into its own L1, and hands them over a batch at a time.
//
// What a batch is and which ones exist is combine_fabric2d_group_walk.hpp; this core takes those where
// batch % num_peers == my_index. The rest of the group takes the others, and since every core of the group
// builds the same walk, that split needs no coordination.
//
// Two counters, the same monotonic single-writer idiom the reader and sender already use between themselves.
// `produced` lives on each consumer's core and is bumped here; `freed[c]` lives here, one per consumer,
// and is bumped there. Per consumer rather than summed: consumers run far apart -- a group's senders take
// alternating halves of each run -- and one sum would let the leading one's credit release a slot the
// trailing one is still reading.
//
// A TILED dispatched buffer is read a block of tiles at a time and untilized by the compute kernel on this
// core, a whole tile-row per batch because that is the least an untilize can produce. A ROW_MAJOR one needs
// no untilize and has no compute kernel, so the rows go straight into the batch and only the ones the walk
// asked for are read at all.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "combine_fabric2d_untilizer_ct_args.hpp"
#include "combine_fabric2d_group_walk.hpp"

constexpr cmbf2d::UntilizerCtArgs ct{};

struct Dram {
    decltype(TensorAccessor(ct.dram_in_args, uint32_t{})) in;
    decltype(TensorAccessor(ct.dram_counts_args, uint32_t{})) counts;
    decltype(TensorAccessor(ct.dram_region_args, uint32_t{})) region;
    decltype(TensorAccessor(ct.dram_expert_offsets_args, uint32_t{})) expert_offsets;
};

Dram open_dram() {
    return Dram{
        TensorAccessor(ct.dram_in_args, ct.dram_in_base_addr),
        TensorAccessor(ct.dram_counts_args, ct.dram_counts_base_addr),
        TensorAccessor(ct.dram_region_args, ct.dram_region_base_addr),
        TensorAccessor(ct.dram_expert_offsets_args, ct.dram_expert_offsets_base_addr)};
}

cmbf2d::ControlTables read_control_tables(const Dram& dram) {
    constexpr uint32_t row_bytes = ct.num_routed_experts * 4;
    cmbf2d::ControlTables ctl{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + ct.dispatch_group_size * row_bytes),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + (ct.dispatch_group_size + 1) * row_bytes),
        ct.num_routed_experts,
        ct.dispatch_group_size};
    for (uint32_t r = 0; r < ct.dispatch_group_size; r++) {
        noc_async_read(dram.expert_offsets.get_noc_addr(r), ct.control_addr + r * row_bytes, row_bytes);
    }
    noc_async_read(dram.counts.get_noc_addr(0), (uint32_t)ctl.counts, row_bytes);
    noc_async_read(dram.region.get_noc_addr(0), (uint32_t)ctl.region, row_bytes);
    noc_async_read_barrier();
    return ctl;
}

cmbf2d::GroupWalk walk_for(const cmbf2d::ControlTables& ctl, uint32_t local_expert) {
    return cmbf2d::group_walk(
        ctl, ct.walks_down != 0, ct.my_expert_base + local_expert, ct.my_dg_index, ct.num_destinations, [](uint32_t k) {
            return kernel_compile_time_args[ct.destination_base + k];
        });
}

uint64_t consumer_noc(uint32_t c, uint32_t addr) {
    const uint32_t w = ct.consumer_base + c * cmbf2d::UNT_PEER_WORDS;
    return get_noc_addr(kernel_compile_time_args[w + 0], kernel_compile_time_args[w + 1], addr);
}

volatile tt_l1_ptr uint32_t* freed_by(uint32_t c) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kernel_compile_time_args[ct.consumer_base + c * cmbf2d::UNT_PEER_WORDS + 2]);
}

// Batches this core has handed over and batches whose slots it has taken back. Kept apart so the next batch
// can be built while the consumers still hold the one before it.
struct Ring {
    CircularBuffer cb_out{cmbf2d::UNT_CB_OUT};
    uint32_t produced = 0;
    uint32_t popped = 0;

    bool full() const { return produced - popped >= ct.ring_batches; }

    // Wait until every consumer has passed the oldest batch still held, then take its slot back. The minimum
    // over consumers rather than their sum: only the slowest one says the slot is really free.
    void reclaim_one() {
        for (uint32_t c = 0; c < ct.num_consumers; c++) {
            volatile tt_l1_ptr uint32_t* freed = freed_by(c);
            invalidate_l1_cache();
            while (*freed < popped + 1) {
                invalidate_l1_cache();
            }
        }
        cb_out.pop_front(cmbf2d::UNT_BATCH_ROWS);
        popped++;
    }

    // Hand the batch over once its rows really exist. The slot is not reclaimed here: that waits until the
    // ring is full, which is what lets the next batch be built while the consumers work through this one.
    void publish() {
        cb_out.wait_front((produced - popped + 1) * cmbf2d::UNT_BATCH_ROWS);
        for (uint32_t c = 0; c < ct.num_consumers; c++) {
            noc_semaphore_inc(consumer_noc(c, ct.produced_addr), 1);
        }
        produced++;
    }
};

// The walk, once per local expert, with `body(batch_in_expert, walk)` called for the batches this core owns.
template <typename Body>
void walk_my_batches(const cmbf2d::ControlTables& ctl, Body body) {
    uint32_t batch = 0;
    for (uint32_t local_expert = 0; local_expert < ct.experts_per_chip; local_expert++) {
        const cmbf2d::GroupWalk walk = walk_for(ctl, local_expert);
        for (uint32_t b = 0; b < walk.num_batches(); b++, batch++) {
            if (batch % ct.num_peers == ct.my_index) {
                body(b, walk);
            }
        }
    }
}

void kernel_main() {
    const Dram dram = open_dram();
    const cmbf2d::ControlTables ctl = read_control_tables(dram);

    // The compute kernel cannot read the control tensors, so it is told how many batches to expect before
    // the first one arrives. Pushed once and never popped.
    uint32_t mine = 0;
    walk_my_batches(ctl, [&](uint32_t, const cmbf2d::GroupWalk&) { mine++; });
    CircularBuffer cb_batches(cmbf2d::UNT_CB_BATCHES);
    cb_batches.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_batches.get_write_ptr()) = mine;
    cb_batches.push_back(1);

    Ring ring;
    walk_my_batches(ctl, [&](uint32_t b, const cmbf2d::GroupWalk& walk) {
        while (ring.full()) {
            ring.reclaim_one();
        }
        // The whole tile-row, a block of tiles at a time so the input window stays small. Whole because that
        // is the least an untilize can do, even when the walk wants only part of it.
        CircularBuffer cb_in(cmbf2d::UNT_CB_IN);
        const uint32_t first_tile = walk.tile_row_of(b) * ct.tiles_per_row;
        for (uint32_t t = 0; t < ct.tiles_per_row; t += ct.block_tiles) {
            cb_in.reserve_back(ct.block_tiles);
            const uint32_t dst = cb_in.get_write_ptr();
            for (uint32_t j = 0; j < ct.block_tiles; j++) {
                noc_async_read(dram.in.get_noc_addr(first_tile + t + j), dst + j * ct.tile_bytes, ct.tile_bytes);
            }
            noc_async_read_barrier();
            cb_in.push_back(ct.block_tiles);
        }
        ring.publish();
    });

    while (ring.popped < ring.produced) {
        ring.reclaim_one();
    }

    // Back to zero for the next launch, which starts its own count at zero. Every batch this core staged has
    // been released by every consumer, so nothing is still counting up.
    for (uint32_t c = 0; c < ct.num_consumers; c++) {
        noc_semaphore_set(freed_by(c), 0);
    }
}
