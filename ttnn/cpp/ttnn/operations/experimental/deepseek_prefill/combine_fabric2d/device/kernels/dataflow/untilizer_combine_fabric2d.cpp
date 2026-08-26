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
// This is the row-major path: the dispatched buffer already holds rows, so a batch is a copy of the pages
// the walk asked for into the slot, at the offset their page index implies. Nothing is untilized yet.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
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

// Batches this core has staged and batches whose slots it has taken back. Kept apart so the next batch can
// be built while the consumers still hold the one before it.
struct Ring {
    uint32_t produced = 0;
    uint32_t popped = 0;

    uint32_t slot_addr() const {
        return ct.ring_addr + (produced % ct.ring_batches) * cmbf2d::UNT_BATCH_ROWS * ct.token_size_bytes;
    }

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
        popped++;
    }

    void publish() {
        for (uint32_t c = 0; c < ct.num_consumers; c++) {
            noc_semaphore_inc(consumer_noc(c, ct.produced_addr), 1);
        }
        produced++;
    }
};

void kernel_main() {
    const Dram dram = open_dram();
    const cmbf2d::ControlTables ctl = read_control_tables(dram);

    Ring ring;
    uint32_t batch = 0;  // position in the group's walk, running across experts
    for (uint32_t local_expert = 0; local_expert < ct.experts_per_chip; local_expert++) {
        const cmbf2d::GroupWalk walk = walk_for(ctl, local_expert);
        for (uint32_t b = 0; b < walk.num_batches(); b++, batch++) {
            if (batch % ct.num_peers != ct.my_index) {
                continue;
            }
            while (ring.produced - ring.popped >= ct.ring_batches) {
                ring.reclaim_one();
            }
            uint32_t lo = 0;
            uint32_t hi = 0;
            walk.batch_pages(b, lo, hi);
            const uint32_t dst = ring.slot_addr();
            for (uint32_t p = lo; p < hi; p++) {
                noc_async_read(
                    dram.in.get_noc_addr(p),
                    dst + (p % cmbf2d::UNT_BATCH_ROWS) * ct.token_size_bytes,
                    ct.token_size_bytes);
            }
            noc_async_read_barrier();
            ring.publish();
        }
    }

    while (ring.popped < ring.produced) {
        ring.reclaim_one();
    }

    // Back to zero for the next launch, which starts its own count at zero. Every batch this core staged has
    // been released by every consumer, so nothing is still counting up.
    for (uint32_t c = 0; c < ct.num_consumers; c++) {
        noc_semaphore_set(freed_by(c), 0);
    }
}
