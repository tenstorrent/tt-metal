// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The untilizer kernel's body, shared by combine_fabric2d and by the routed expert's overlapped fork.
//
// Included after the owning op's own compile-time arguments: the includer supplies `ct`, the alias
// `cmbf2d_ns`, and the ring-counter accessors. CMBF2D_OVERLAPPED guards the two things that genuinely
// differ -- which expert a walk step names, and the gate on the routed expert having written it.

#pragma once

// The includer must already have pulled in its own untilizer rt args and group walk.

constexpr cmbf2d_ns::UntilizerCtArgs ct{};

struct Dram {
    decltype(TensorAccessor(ct.dram_in_args, uint32_t{})) in;
    decltype(TensorAccessor(ct.dram_counts_args, uint32_t{})) counts;
    decltype(TensorAccessor(ct.dram_region_args, uint32_t{})) region;
    decltype(TensorAccessor(ct.dram_expert_offsets_args, uint32_t{})) expert_offsets;
#ifdef CMBF2D_OVERLAPPED
    decltype(TensorAccessor(ct.dram_expert_table_args, uint32_t{})) expert_table;
#endif
};

Dram open_dram() {
    const auto rt = cmbf2d_ns::UntilizerRtArgManager::get_rt_args();
    return Dram{
        TensorAccessor(ct.dram_in_args, rt.dram_in),
        TensorAccessor(ct.dram_counts_args, rt.dram_counts),
        TensorAccessor(ct.dram_region_args, rt.dram_region),
        TensorAccessor(ct.dram_expert_offsets_args, rt.dram_expert_offsets)
#ifdef CMBF2D_OVERLAPPED
            ,
        TensorAccessor(ct.dram_expert_table_args, rt.dram_expert_table)
#endif
    };
}

cmbf2d_ns::ControlTables read_control_tables(const Dram& dram) {
    constexpr uint32_t row_bytes = ct.num_routed_experts * 4;
#ifdef CMBF2D_OVERLAPPED
    constexpr uint32_t ids_addr = ct.control_addr + cmbf2d_ns::align_control((ct.dispatch_group_size + 2) * row_bytes);
    constexpr uint32_t ids_stride = cmbf2d_ns::expert_table_row_stride(ct.experts_per_chip);
#endif
    cmbf2d_ns::ControlTables ctl{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + ct.dispatch_group_size * row_bytes),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + (ct.dispatch_group_size + 1) * row_bytes),
        ct.num_routed_experts,
        ct.dispatch_group_size,
#ifdef CMBF2D_OVERLAPPED
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_addr),
        ids_stride / 4,
#endif
    };
    for (uint32_t r = 0; r < ct.dispatch_group_size; r++) {
        noc_async_read(dram.expert_offsets.get_noc_addr(r), ct.control_addr + r * row_bytes, row_bytes);
#ifdef CMBF2D_OVERLAPPED
        noc_async_read(
            dram.expert_table.get_noc_addr(ct.expert_table_page_base + r),
            ids_addr + r * ids_stride,
            ct.experts_per_chip * 4);
#endif
    }
    noc_async_read(dram.counts.get_noc_addr(0), (uint32_t)ctl.counts, row_bytes);
    noc_async_read(dram.region.get_noc_addr(0), (uint32_t)ctl.region, row_bytes);
    noc_async_read_barrier();
    return ctl;
}

cmbf2d_ns::GroupWalk walk_for(const cmbf2d_ns::ControlTables& ctl, uint32_t step) {
#ifdef CMBF2D_OVERLAPPED
    // Overlapped, steps are the routed expert's threshold-split order, not this chip's slot order; the
    // group's readers build the same walk from the same tables.
    const uint32_t expert =
        cmbf2d_ns::expert_at_step(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, step);
#else
    const uint32_t expert = ct.my_expert_base + step;
#endif
    return cmbf2d_ns::group_walk(ctl, ct.walks_down != 0, expert, ct.my_dg_index, ct.num_destinations, [](uint32_t k) {
        return kernel_compile_time_args[ct.destination_base + k];
    });
}

uint64_t consumer_noc(uint32_t c, uint32_t addr) {
    const uint32_t w = ct.consumer_base + c * cmbf2d_ns::UNT_PEER_WORDS;
    return get_noc_addr(kernel_compile_time_args[w + 0], kernel_compile_time_args[w + 1], addr);
}

// The peer word names consumer c's `freed` count on this core; what it names is the owning op's choice.
volatile tt_l1_ptr uint32_t* freed_by(uint32_t c) {
    return ct.freed_ptr(kernel_compile_time_args[ct.consumer_base + c * cmbf2d_ns::UNT_PEER_WORDS + 2]);
}

// Batches this core has handed over and batches whose slots it has taken back. Kept apart so the next batch
// can be built while the consumers still hold the one before it.
struct Ring {
    CircularBuffer cb_out{cmbf2d_ns::UNT_CB_OUT};
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
        cb_out.pop_front(cmbf2d_ns::UNT_BATCH_ROWS);
        popped++;
    }

    // Hand the batch over once its rows really exist. The slot is not reclaimed here: that waits until the
    // ring is full, which is what lets the next batch be built while the consumers work through this one.
    void publish() {
        cb_out.wait_front((produced - popped + 1) * cmbf2d_ns::UNT_BATCH_ROWS);
        for (uint32_t c = 0; c < ct.num_consumers; c++) {
            noc_semaphore_inc(consumer_noc(c, ct.produced_addr_value()), 1);
        }
        produced++;
    }
};

// The walk, once per walk step, with `on_step(step)` called before a step's batches and
// `body(batch_in_expert, walk)` for the batches this core owns.
template <typename OnStep, typename Body>
void walk_my_batches(const cmbf2d_ns::ControlTables& ctl, OnStep on_step, Body body) {
    uint32_t batch = 0;
    for (uint32_t step = 0; step < ct.experts_per_chip; step++) {
        on_step(step);
        const cmbf2d_ns::GroupWalk walk = walk_for(ctl, step);
        for (uint32_t b = 0; b < walk.num_batches(); b++, batch++) {
            if (batch % ct.num_peers == ct.my_index) {
                body(b, walk);
            }
        }
    }
}

void kernel_main() {
    const Dram dram = open_dram();
    const cmbf2d_ns::ControlTables ctl = read_control_tables(dram);

    // The compute kernel cannot read the control tensors, so it is told how many batches to expect before
    // the first one arrives. Pushed once and never popped.
    uint32_t mine = 0;
#ifndef CMBF2D_IDLE
    walk_my_batches(ctl, [](uint32_t) {}, [&](uint32_t, const cmbf2d_ns::GroupWalk&) { mine++; });
#endif
    CircularBuffer cb_batches(cmbf2d_ns::UNT_CB_BATCHES);
    cb_batches.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_batches.get_write_ptr()) = mine;
    cb_batches.push_back(1);
#ifdef CMBF2D_IDLE
    // Measurement mode: zero batches announced, so the compute kernel exits at once.
    return;
#endif

#ifdef CMBF2D_OVERLAPPED
    // A step's tile-rows are the routed expert's output: wait until it reports them written. Only here,
    // not in the counting walk above, which must finish before any expert is ready.
    const auto on_step = [&](uint32_t step) {
        const uint32_t local =
            cmbf2d_ns::local_at_step(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, step);
        cmbf2d_ns::wait_for_ready(
            ct.ready_sem,
            cmbf2d_ns::ready_target(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, local));
    };
#else
    const auto on_step = [](uint32_t) {};
#endif

    Ring ring;
    walk_my_batches(ctl, on_step, [&](uint32_t b, const cmbf2d_ns::GroupWalk& walk) {
        while (ring.full()) {
            ring.reclaim_one();
        }
        // The whole tile-row, a block of tiles at a time so the input window stays small. Whole because that
        // is the least an untilize can do, even when the walk wants only part of it.
        CircularBuffer cb_in(cmbf2d_ns::UNT_CB_IN);
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

    // Every batch this core staged has been released by every consumer, so nothing is still counting up.
    // Whether the counts need zeroing for the next launch depends on where they live.
    for (uint32_t c = 0; c < ct.num_consumers; c++) {
        ct.reset_freed_counter(freed_by(c));
    }
}
