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
#include "combine_fabric2d_untilizer_rt_args.hpp"
#include "combine_fabric2d_group_walk.hpp"

// CMBF2D_OVERLAPPED guards the two things that genuinely differ between this op and the routed expert's
// overlapped build -- which expert a walk step names, and the gate on the routed expert having written it.

constexpr ::cmbf2d::UntilizerCtArgs ct{};

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
    const auto rt = ::cmbf2d::UntilizerRtArgManager::get_rt_args();
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

::cmbf2d::ControlTables read_control_tables(const Dram& dram) {
    constexpr uint32_t row_bytes = ct.num_routed_experts * 4;
#ifdef CMBF2D_OVERLAPPED
    constexpr uint32_t ids_addr = ct.control_addr + ::cmbf2d::align_control((ct.dispatch_group_size + 2) * row_bytes);
    constexpr uint32_t ids_stride = ::cmbf2d::expert_table_row_stride(ct.experts_per_chip);
#endif
    ::cmbf2d::ControlTables ctl{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + ct.dispatch_group_size * row_bytes),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.control_addr + (ct.dispatch_group_size + 1) * row_bytes),
        ct.num_routed_experts,
        ct.dispatch_group_size,
        ct.dispatch_capacity,
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

::cmbf2d::GroupWalk walk_for(const ::cmbf2d::ControlTables& ctl, uint32_t step) {
#ifdef CMBF2D_OVERLAPPED
    // Overlapped, steps are the routed expert's threshold-split order, not this chip's slot order; the
    // group's readers build the same walk from the same tables.
    const uint32_t expert =
        ::cmbf2d::expert_at_step(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, step);
#else
    const uint32_t expert = ct.my_expert_base + step;
#endif
    return ::cmbf2d::group_walk(ctl, ct.walks_down != 0, expert, ct.my_dg_index, ct.num_destinations, [](uint32_t k) {
        return kernel_compile_time_args[ct.destination_base + k];
    });
}

uint64_t consumer_noc(uint32_t c, uint32_t addr) {
    const uint32_t w = ct.consumer_base + c * ::cmbf2d::UNT_PEER_WORDS;
    return get_noc_addr(kernel_compile_time_args[w + 0], kernel_compile_time_args[w + 1], addr);
}

// The peer word names consumer c's `freed` count on this core; what it names is the owning op's choice.
volatile tt_l1_ptr uint32_t* freed_by(uint32_t c) {
    return ct.freed_ptr(kernel_compile_time_args[ct.consumer_base + c * ::cmbf2d::UNT_PEER_WORDS + 2]);
}

// Batches this core has handed over and batches whose slots it has taken back. Kept apart so the next batch
// can be built while the consumers still hold the one before it.
struct Ring {
    CircularBuffer cb_out{::cmbf2d::UNT_CB_OUT};
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
        cb_out.pop_front(::cmbf2d::UNT_BATCH_ROWS);
        popped++;
    }

    // Hand the batch over once its rows really exist. The slot is not reclaimed here: that waits until the
    // ring is full, which is what lets the next batch be built while the consumers work through this one.
    void publish() {
        cb_out.wait_front((produced - popped + 1) * ::cmbf2d::UNT_BATCH_ROWS);
        for (uint32_t c = 0; c < ct.num_consumers; c++) {
            noc_semaphore_inc(consumer_noc(c, ct.produced_slot_addr()), 1);
        }
        produced++;
    }
};

#ifdef CMBF2D_FP8_ROWS
// fp8 tokens: the compute kernel packs a batch's e4m3 values, and their scales are this kernel's to write after them
// in each row, from the 1/scale tile it leaves per scale block. Column 0 of that tile holds one exact power of two per
// row, so the fp32 scale is (254 << 23) minus its bits.
struct ScaleTails {
    static constexpr uint32_t block_tiles = 4;
    static constexpr uint32_t values_bytes = ct.tiles_per_row * 32;  // one byte per e4m3 value
    CircularBuffer cb{::cmbf2d::UNT_CB_SCALE};
    // cb_out's first byte, where batch n's rows start (n % ring_batches) batches in: the compute kernel fills the
    // ring a whole batch at a time.
    uint32_t ring_base = CircularBuffer(::cmbf2d::UNT_CB_OUT).get_read_ptr();
    uint32_t rows = 0;
    uint32_t next_block = 0;

    void start(uint32_t batch) {
        rows = ring_base + batch % ct.ring_batches * ::cmbf2d::UNT_BATCH_ROWS * ct.token_size_bytes;
        next_block = 0;
    }

    // The next `n` scale blocks' scales, once the compute kernel has them.
    void write(uint32_t n) {
        cb.wait_front(n);
        invalidate_l1_cache();
        const uint32_t tile_bytes = get_tile_size(::cmbf2d::UNT_CB_SCALE);
        for (uint32_t i = 0; i < n; i++) {
            const uint16_t* inv = reinterpret_cast<const uint16_t*>(cb.get_read_ptr() + i * tile_bytes);
            const uint32_t tail = rows + values_bytes + (next_block + i) * sizeof(uint32_t);
            for (uint32_t r = 0; r < ::cmbf2d::UNT_BATCH_ROWS; r++) {
                // Rows 0-15 of column 0 are in face 0 and rows 16-31 in face 2, 16 values to a face row.
                const uint32_t v = inv[(r / 16) * 512 + (r % 16) * 16];
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tail + r * ct.token_size_bytes) =
                    (254u << 23) - (v << 16);
            }
        }
        cb.pop_front(n);
        next_block += n;
    }
};
#endif

// The walk, once per walk step, with `on_step(step)` called before a step's batches and
// `body(batch_in_expert, walk)` for the batches this core owns.
template <typename OnStep, typename Body>
void walk_my_batches(const ::cmbf2d::ControlTables& ctl, OnStep on_step, Body body) {
    uint32_t batch = 0;
    for (uint32_t step = 0; step < ct.experts_per_chip; step++) {
        on_step(step);
        const ::cmbf2d::GroupWalk walk = walk_for(ctl, step);
        for (uint32_t b = 0; b < walk.num_batches(); b++, batch++) {
            if (batch % ct.num_peers == ct.my_index) {
                body(b, walk);
            }
        }
    }
}

void kernel_main() {
    const Dram dram = open_dram();
    const ::cmbf2d::ControlTables ctl = read_control_tables(dram);

    // The compute kernel cannot read the control tensors, so it is told how many batches to expect before
    // the first one arrives. Pushed once and never popped.
    uint32_t mine = 0;
    walk_my_batches(ctl, [](uint32_t) {}, [&](uint32_t, const ::cmbf2d::GroupWalk&) { mine++; });
    CircularBuffer cb_batches(::cmbf2d::UNT_CB_BATCHES);
    cb_batches.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_batches.get_write_ptr()) = mine;
    cb_batches.push_back(1);

#ifdef CMBF2D_OVERLAPPED
    // A step's tile-rows are the routed expert's output: wait until it reports them written. Only here,
    // not in the counting walk above, which must finish before any expert is ready.
#ifdef CMBF2D_PROGRESS_SEM
    // A unified-pass expert is written chunk by chunk, so its tile-rows are taken as the collector reports
    // them (`progress`), not all at once at the end; the per-row gate is in the batch loop below.
    bool gate_rows = false;
    uint32_t gate_step = 0;
    uint32_t gate_target = 0;
    uint32_t gate_base_row = 0;
#endif
    const auto on_step = [&](uint32_t step) {
        const uint32_t local =
            ::cmbf2d::local_at_step(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, step);
        const uint32_t target =
            ::cmbf2d::ready_target(ctl, ct.my_dg_index, ct.experts_per_chip, ct.expert_threshold, local);
#ifdef CMBF2D_PROGRESS_SEM
        gate_rows = !::cmbf2d::in_fused_pass(ctl, ct.my_dg_index, ct.expert_threshold, local);
        if (gate_rows) {
            // The collector counts the routed expert's steps, one per slot walked in each pass, not this
            // walk's steps: the one whose completion makes `ready` reach target is target - 1.
            gate_step = target - 1;
            gate_target = target;
            gate_base_row = ctl.region[ctl.expert_id(ct.my_dg_index, local)] / ::cmbf2d::UNT_BATCH_ROWS;
            return;
        }
#endif
        ::cmbf2d::wait_for_ready(ct.ready_sem, target);
    };
#else
    const auto on_step = [](uint32_t) {};
#endif

    Ring ring;
#ifdef CMBF2D_FP8_ROWS
    ScaleTails tails;
    constexpr uint32_t window_blocks = ct.block_tiles / ScaleTails::block_tiles;
#endif
    walk_my_batches(ctl, on_step, [&](uint32_t b, const ::cmbf2d::GroupWalk& walk) {
        while (ring.full()) {
            ring.reclaim_one();
        }
        // The whole tile-row, a block of tiles at a time so the input window stays small. Whole because that
        // is the least an untilize can do, even when the walk wants only part of it.
#ifdef CMBF2D_PROGRESS_SEM
        if (gate_rows) {
            // Either the whole step is out, or the chunk holding this tile-row is.
            volatile tt_l1_ptr uint32_t* ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ct.ready_sem));
            volatile tt_l1_ptr uint32_t* progress =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CMBF2D_PROGRESS_SEM));
            const uint32_t rel_row = walk.tile_row_of(b) - gate_base_row;
            const auto released = [&]() {
                const uint32_t p = *progress;
                if ((p >> 24) != gate_step) {
                    return (p >> 24) > gate_step;  // a later step's chunks only start once this step is out
                }
                const uint32_t rows = (p >> 16) & 0xFF;
                return rows != 0 && rel_row / rows < 16 && (p & (1u << (rel_row / rows))) != 0;
            };
            invalidate_l1_cache();
            while (*ready < gate_target && !released()) {
                invalidate_l1_cache();
            }
        }
#endif
        CircularBuffer cb_in(::cmbf2d::UNT_CB_IN);
        const uint32_t first_tile = walk.tile_row_of(b) * ct.tiles_per_row;
#ifdef CMBF2D_FP8_ROWS
        tails.start(ring.produced);
#endif
        for (uint32_t t = 0; t < ct.tiles_per_row; t += ct.block_tiles) {
            cb_in.reserve_back(ct.block_tiles);
            const uint32_t dst = cb_in.get_write_ptr();
            for (uint32_t j = 0; j < ct.block_tiles; j++) {
                noc_async_read(dram.in.get_noc_addr(first_tile + t + j), dst + j * ct.tile_bytes, ct.tile_bytes);
            }
            noc_async_read_barrier();
            cb_in.push_back(ct.block_tiles);
#ifdef CMBF2D_FP8_ROWS
            // A window behind the reads, so the compute kernel quantizes this window while the last one's scales go in.
            if (t != 0) {
                tails.write(window_blocks);
            }
#endif
        }
#ifdef CMBF2D_FP8_ROWS
        tails.write(window_blocks);
#endif
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
