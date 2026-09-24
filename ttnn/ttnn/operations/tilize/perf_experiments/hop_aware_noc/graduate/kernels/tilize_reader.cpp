// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0) — the `load_block` block operation (op_design.md).
//
// Custom block operation, not dataflow_kernel_lib::read_sticks_for_tilize: that
// helper derives BOTH the CB push quantum and the L1 stick stride from the valid
// row bytes, so narrowing a ragged last column block would push fewer than
// block_width pages and break the CB ring-wrap invariant
// (tilize_helpers_dataflow.inl:92-93, 117, 127; see op_design.md API Mapping).
//
// Per Tensix core: output-tile rectangle [row_start, row_start + core_row_tiles)
// x [col_start, col_start + core_col_tiles), cut into column blocks of
// block_width tiles. Per CB quantum of rows_per_quantum walk positions (tile-rows):
// rows_per_quantum * tile_h stick-segment reads in flight, one barrier, one push
// of the nominal rows_per_quantum * block_width pages (only the NoC transfer
// narrows on the ragged last column block; only the kernel's final quantum may
// hold fewer tile-rows). With read_ahead > 1 the next quantum's reads are issued
// before the previous quantum's (transaction-id) barrier.
//
// Split reader (CT `split_reader`): this RISC-V produces only the EVEN walk
// positions; the BRISC writer produces the odd ones into its own CB.
//
// Padded (CT `padded`, output_padded_shape / pad_value beyond the input): the walk is over
// the PADDED tile grid; each tile-row's sticks come from the per-image stick map
// (tilize_stick_reads.hpp PadMap) and everything outside the input is filled with the pad
// value (PadFill: fill_l1_range stores for short ranges, NoC loopback copies from a
// pre-filled source for long ones), all before the slot is pushed.
//
// Co-read (CT `co_read`, Refinement 8): when every Tensix core's walk is ONE position, this
// RISC-V issues the first tile_h - co_read stick reads of the tile-row and the writer RISC-V
// (BRISC, NoC1) the rest into the same reserved slot; the slot is pushed after this RISC-V's
// read barrier AND the writer's landed flag (tilize_stick_reads.hpp CoReadLanded).
//
// Resident input (CT `input_resident`, sharded_resident regime): cb_input_sticks
// is backed on this Tensix core's own input shard, which already holds the
// tilize stick layout, so load_block is a publish of the shard's pages: no NoC
// read at all.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tilize_stick_reads.hpp"

// retile_l1_facewalk load_block (tilize_stick_reads.hpp): walk the core's rectangle in
// units of row_align output tile-rows; per unit, get its input tiles into L1 (a
// page-sized NoC read per tile into the staging ring with up to stage_depth - 1 units
// in flight ahead of the face walk, or in place when the input shard is resident),
// then face-walk them into cb_input_sticks tile-row by tile-row.
template <
    uint32_t cb_input_sticks,
    uint32_t cb_retile_staging,
    uint32_t block_width,
    uint32_t tile_h,
    uint32_t in_tile_h,
    uint32_t tile_col_bytes,
    uint32_t in_page_bytes,
    uint32_t rows_per_quantum,
    uint32_t row_align,
    uint32_t stage_depth,
    bool source_resident,
    bool facewalk_noc,
    typename Accessor>
FORCE_INLINE void read_retile(
    const Accessor& accessor,
    uint32_t row_start,
    uint32_t core_row_tiles,
    uint32_t col_start,
    uint32_t core_col_tiles,
    uint32_t row_rotation,
    uint32_t tiles_per_row,
    uint32_t out_rows_per_image,
    uint32_t in_rows_per_image) {
    using namespace tilize_dataflow;
    static_assert(stage_depth >= 1 && stage_depth <= 14, "one NoC transaction id per staging slot + the face walk's");
    constexpr uint32_t unit_in_rows = tile_h > in_tile_h ? tile_h / in_tile_h : 1;
    constexpr uint32_t stage_slot_bytes = unit_in_rows * block_width * in_page_bytes;
    constexpr uint32_t in_tile_bytes = tile_h * tile_col_bytes;
    // Staging reads use trids 1..stage_depth; the face walk's loopback reads use the next one.
    using Walk = FaceWalk<in_tile_h, tile_h, block_width, tile_col_bytes, in_page_bytes, facewalk_noc, stage_depth + 1>;
    RowSlotWriter<cb_input_sticks, block_width, rows_per_quantum, block_width * in_tile_bytes> out;

    // The host keeps row_start, core_row_tiles and row_rotation multiples of row_align, so a
    // unit never wraps inside the rotated walk and the writer's walk visits the same order.
    Walker<block_width> walk(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
    const uint32_t num_units = walk.num_positions() / row_align;
    auto next_unit = [&]() {
        RetileUnit u{walk.row(), walk.first_col(), walk.valid_width(), 0, 0};
        retile_source_of(u.row, tile_h, in_tile_h, out_rows_per_image, in_rows_per_image, u.first_in_row, u.h_off);
        for (uint32_t k = 0; k < row_align; ++k) {
            walk.advance();
        }
        return u;
    };
    // Face-walk one unit. On return every face row has landed, so its source slot may be refilled
    // and every pushed cb_input_sticks slot holds its data.
    auto face_walk_unit = [&](const RetileUnit& u, uint32_t src) {
        Walk::begin();
        for (uint32_t j = 0; j < row_align; ++j) {
            Walk::tile_row(out.open_row(), src, u.h_off + j * tile_h, u.valid_width);
            if (out.row_ends_slot()) {
                Walk::complete();
            }
            out.close_row();
        }
        Walk::complete();
    };

    if constexpr (source_resident) {
        // cb_retile_staging IS this core's input shard ([shard tile-rows] x block_width tiles,
        // tile-row-major): the unit's tiles are read in place, no NoC traffic.
        const uint32_t shard_base = get_read_ptr(cb_retile_staging);
        uint32_t shard_first_in_row, unused_h_off;
        retile_source_of(
            row_start, tile_h, in_tile_h, out_rows_per_image, in_rows_per_image, shard_first_in_row, unused_h_off);
        for (uint32_t k = 0; k < num_units; ++k) {
            const RetileUnit u = next_unit();
            face_walk_unit(
                u,
                shard_base +
                    ((u.first_in_row - shard_first_in_row) * block_width + (u.first_col - col_start)) * in_page_bytes);
        }
    } else {
        const uint32_t stage_base = get_write_ptr(cb_retile_staging);
        RetileUnit pending[stage_depth];
        auto issue = [&](uint32_t k) {
            asm volatile("" ::: "memory");  // the slot's earlier face-walk loads precede its overwrite
            const RetileUnit u = next_unit();
            pending[k % stage_depth] = u;
            const uint32_t slot = stage_base + (k % stage_depth) * stage_slot_bytes;
            noc_async_read_set_trid(1 + (k % stage_depth));
            for (uint32_t i = 0; i < unit_in_rows; ++i) {
                const uint32_t first_tile = (u.first_in_row + i) * tiles_per_row + u.first_col;
                const uint32_t l1_row = slot + i * block_width * in_page_bytes;
                for (uint32_t c = 0; c < u.valid_width; ++c) {
                    noc_async_read(accessor.get_noc_addr(first_tile + c), l1_row + c * in_page_bytes, in_page_bytes);
                }
            }
        };
        const uint32_t prefetch = num_units < stage_depth - 1 ? num_units : stage_depth - 1;
        for (uint32_t k = 0; k < prefetch; ++k) {
            issue(k);
        }
        for (uint32_t k = 0; k < num_units; ++k) {
            if (k + stage_depth - 1 < num_units) {
                issue(k + stage_depth - 1);  // lands in the slot unit k - 1 has already been walked out of
            }
            noc_async_read_barrier_with_trid(1 + (k % stage_depth));
            asm volatile("" ::: "memory");  // the face walk's L1 loads must follow the barrier
            face_walk_unit(pending[k % stage_depth], stage_base + (k % stage_depth) * stage_slot_bytes);
        }
        noc_async_read_set_trid(0);
    }
    out.finish();
}

// bank_coalesced load_block (Refinement 6): the stick reader for a DRAM-interleaved input whose
// every Tensix core reads WHOLE sticks in one column block (host-gated). Stick page p lives in
// bank p % num_banks at offset (p / num_banks) * stick_page_bytes, so the sticks of one run of
// consecutive tile-rows that share a bank are contiguous in that bank: one NoC read per bank
// fetches all of them (~count / num_banks sticks instead of one 128-byte read per stick).
//
// Per CB quantum (rows_per_quantum walk positions = a unit): its runs of consecutive tile-rows
// (a rotated walk wraps at most once inside a unit) are read bank by bank into a reader-private
// staging ring slot (bank-major: bank j's sticks first + j, first + j + NB, ... back to back),
// with up to stage_depth - 1 units in flight ahead. Once a unit has landed, NoC loopback reads
// (one packet per stick, the RISC-V only issues commands, as in FaceWalk) move each stick to
// its tilize position in the reserved cb_input_sticks slot; after their barrier the slot is
// pushed. Compute and the writer see the same slots, in the same walk order, as StickProducer's.
//
// Bank rotation: each core starts its per-run bank loop at bank_rotation % banks, so the 64
// cores' concurrent requests spread over the banks.
template <
    uint32_t cb_input_sticks,
    uint32_t cb_staging,
    uint32_t block_width,
    uint32_t tile_h,
    uint32_t stick_bytes,       // L1 bytes of one stick in cb_input_sticks (the whole stick's data)
    uint32_t stick_page_bytes,  // aligned interleaved page stride (staging stride too)
    uint32_t rows_per_quantum,
    uint32_t stage_depth,
    uint32_t num_banks,
    bool scatter_write,
    typename Accessor>
FORCE_INLINE void read_bank_coalesced(
    const Accessor& accessor,
    uint32_t row_start,
    uint32_t core_row_tiles,
    uint32_t col_start,
    uint32_t core_col_tiles,
    uint32_t row_rotation,
    uint32_t bank_rotation) {
    static_assert(stage_depth >= 1 && stage_depth <= 14, "one NoC transaction id per staging slot + the scatter's");
    static_assert(stick_bytes <= stick_page_bytes, "a stick's data fits its page");
    constexpr uint32_t scatter_trid = stage_depth + 1;
    constexpr uint32_t stage_slot_bytes = rows_per_quantum * tile_h * stick_page_bytes;
    constexpr uint32_t slot_pages = rows_per_quantum * block_width;

    // Bank j of a run of `count` sticks: count / NB sticks, one more for j < count % NB; its
    // sticks sit in staging after those of banks 0 .. j - 1.
    struct Run {
        uint32_t banks, q, rem;
        explicit Run(uint32_t count) :
            banks(count < num_banks ? count : num_banks), q(count / num_banks), rem(count % num_banks) {}
        uint32_t sticks(uint32_t j) const { return q + (j < rem ? 1 : 0); }
        uint32_t offset(uint32_t j) const { return j * q + (j < rem ? j : rem); }
    };

    // Visit the runs of consecutive tile-rows of the next `n` walk positions:
    // fn(first tile-row, tile-rows in run, unit position of the run's first tile-row).
    auto for_each_run = [](tilize_dataflow::Walker<block_width>& w, uint32_t n, auto&& fn) {
        uint32_t p = 0;
        while (p < n) {
            const uint32_t run_row = w.row();
            uint32_t run_len = 1;
            w.advance();
            while (p + run_len < n && w.row() == run_row + run_len) {
                ++run_len;
                w.advance();
            }
            fn(run_row, run_len, p);
            p += run_len;
        }
    };

    tilize_dataflow::Walker<block_width> issue_walk(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
    tilize_dataflow::Walker<block_width> scatter_walk(
        row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
    const uint32_t num_positions = issue_walk.num_positions();
    const uint32_t num_units = (num_positions + rows_per_quantum - 1) / rows_per_quantum;
    auto unit_rows = [&](uint32_t k) {
        const uint32_t left = num_positions - k * rows_per_quantum;
        return left < rows_per_quantum ? left : rows_per_quantum;
    };
    const uint32_t stage_base = get_write_ptr(cb_staging);

    auto issue = [&](uint32_t k) {
        MaybeDeviceZoneScope("reader_issue");  // per-bank coalesced DRAM read issue for unit k
        const uint32_t stage = stage_base + (k % stage_depth) * stage_slot_bytes;
        noc_async_read_set_trid(1 + (k % stage_depth));
        for_each_run(issue_walk, unit_rows(k), [&](uint32_t run_row, uint32_t run_len, uint32_t p) {
            const uint32_t first_stick = run_row * tile_h;
            const Run run(run_len * tile_h);
            const uint32_t run_stage = stage + p * tile_h * stick_page_bytes;
            uint32_t j = bank_rotation % run.banks;
            for (uint32_t b = 0; b < run.banks; ++b) {
                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);
                if (++j == run.banks) {
                    j = 0;
                }
            }
        });
    };

    auto scatter = [&](uint32_t k) {
        const uint32_t n = unit_rows(k);
        const uint32_t stage = stage_base + (k % stage_depth) * stage_slot_bytes;
        {
            MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
            cb_reserve_back(cb_input_sticks, slot_pages);
        }
        MaybeDeviceZoneScope("reader_scatter");  // loopback scatter issue + its barrier
        const uint32_t slot = get_write_ptr(cb_input_sticks);
        if constexpr (scatter_write) {
            noc_async_write_one_packet_set_state(get_noc_addr(0), stick_bytes);
        } else {
            noc_async_read_set_trid(scatter_trid);
            noc_async_read_one_packet_set_state(get_noc_addr(0), stick_bytes);
        }
        for_each_run(scatter_walk, n, [&](uint32_t, uint32_t run_len, uint32_t p) {
            const Run run(run_len * tile_h);
            uint32_t src = stage + p * tile_h * stick_page_bytes;
            for (uint32_t j = 0; j < run.banks; ++j) {
                uint32_t dst = slot + (p * tile_h + j) * stick_bytes;
                for (uint32_t i = run.sticks(j); i > 0; --i) {
                    if constexpr (scatter_write) {
                        noc_async_write_one_packet_with_state(src, dst);
                    } else {
                        noc_async_read_one_packet_with_state(src, dst);
                    }
                    src += stick_page_bytes;
                    dst += num_banks * stick_bytes;
                }
            }
        });
        if constexpr (scatter_write) {
            noc_async_write_barrier();
        } else {
            noc_async_read_barrier_with_trid(scatter_trid);
        }
        cb_push_back(cb_input_sticks, n * block_width);
    };

    const uint32_t prefetch = num_units < stage_depth - 1 ? num_units : stage_depth - 1;
    for (uint32_t k = 0; k < prefetch; ++k) {
        issue(k);
    }
    for (uint32_t k = 0; k < num_units; ++k) {
        if (k + stage_depth - 1 < num_units) {
            issue(k + stage_depth - 1);  // lands in the slot unit k - 1 was scattered out of
        }
        {
            MaybeDeviceZoneScope("reader_barrier");  // unit k's DRAM reads landing
            noc_async_read_barrier_with_trid(1 + (k % stage_depth));
        }
        scatter(k);
    }
    noc_async_read_set_trid(0);
}

void kernel_main() {
#ifdef TILIZE_HOP_WRITE_MIN_SAVING
    // Hop-aware write NoC: runs the NoC0 counter re-sync on every return path (tilize_stick_reads.hpp).
    const tilize_dataflow::HopReaderResync hop_resync(get_semaphore(TILIZE_HOP_SEM));
#endif
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);       // tiles per column block (CB quantum)
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);            // sticks per tile-row
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(3);    // bytes of one stick per tile-column
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(4);  // aligned interleaved stick page
    constexpr bool split_reader = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t depth_in = get_compile_time_arg_val(6);          // CB slots (quanta)
    constexpr uint32_t read_ahead = get_compile_time_arg_val(7);        // quanta of reads in flight
    constexpr uint32_t rows_per_quantum = get_compile_time_arg_val(8);  // tile-rows per CB quantum
    constexpr bool input_resident = get_compile_time_arg_val(9) != 0;   // cb_input_sticks backed on the shard
    constexpr uint32_t page_bytes = get_compile_time_arg_val(10);       // data bytes per input page
    constexpr uint32_t pages_per_stick = get_compile_time_arg_val(11);  // input pages per logical stick
    constexpr uint32_t in_tile_h = get_compile_time_arg_val(12);        // 0: ROW_MAJOR input; else retile
    constexpr uint32_t row_align = get_compile_time_arg_val(13);        // retile: output tile-rows per unit
    constexpr uint32_t cb_retile_staging = get_compile_time_arg_val(14);
    constexpr uint32_t retile_stage_depth = get_compile_time_arg_val(15);    // retile: staged units (ring slots)
    constexpr bool retile_facewalk_noc = get_compile_time_arg_val(16) != 0;  // retile: face rows moved by NoC
    constexpr uint32_t read_noc_split = get_compile_time_arg_val(17);        // parked: 0, else other-NoC stick period
    constexpr bool eager_publish = get_compile_time_arg_val(18) != 0;        // parked: push landed slots early
    constexpr uint32_t bank_stride = get_compile_time_arg_val(19);    // parked: 0 off, 1 bank-stride, 2 bank-major
    constexpr bool padded = get_compile_time_arg_val(20) != 0;        // output padded shape > input: fill
    constexpr uint32_t cb_pad_source = get_compile_time_arg_val(21);  // padded: reader-private fill source
    constexpr uint32_t pad_source_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t pad_noc_min_bytes = get_compile_time_arg_val(23);        // shorter fills are CPU stores
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(24);               // input element size
    constexpr bool w_tail_persist = get_compile_time_arg_val(25) != 0;          // padded: band-fill first pass only
    constexpr uint32_t coalesce_depth = get_compile_time_arg_val(26);           // 0, else bank_coalesced staging units
    constexpr bool coalesce_scatter_write = get_compile_time_arg_val(27) != 0;  // loopback writes, else reads
    constexpr uint32_t co_read = get_compile_time_arg_val(28);      // 0, else sticks per tile-row BRISC reads
    constexpr uint32_t co_read_sem = get_compile_time_arg_val(29);  // co-read: the landed flag's semaphore id
    constexpr auto input_args = TensorAccessorArgs<30>();
#ifdef CO_READ_LISTED
    constexpr bool co_read_listed = co_read != 0;  // Perf 2: the host's geometric stick lists
#else
    constexpr bool co_read_listed = false;
#endif
    static_assert(!padded || !split_reader, "the split reader has no pad path");
    static_assert(
        coalesce_depth == 0 || (!padded && !split_reader && pages_per_stick == 1),
        "bank_coalesced: whole unpadded sticks");
    static_assert(!padded || depth_in + 1 <= 15, "the fill's transaction id follows the slots' 1..depth_in");
    static_assert(co_read == 0 || (!padded && !split_reader && coalesce_depth == 0), "co-read: the plain stick walk");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t row_rotation = get_arg_val<uint32_t>(5);
    const uint32_t stick_rotation = get_arg_val<uint32_t>(6);

    if constexpr (in_tile_h != 0) {
        MaybeDeviceZoneScope("reader_retile");  // whole retile load_block (staging reads + face walk)
        read_retile<
            cb_input_sticks,
            cb_retile_staging,
            block_width,
            tile_h,
            in_tile_h,
            tile_col_bytes,
            page_bytes,
            rows_per_quantum,
            row_align,
            retile_stage_depth,
            input_resident,
            retile_facewalk_noc>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            row_rotation,
            get_arg_val<uint32_t>(7),   // tiles_per_row: C of the input tile grid
            get_arg_val<uint32_t>(8),   // output tile-rows per image
            get_arg_val<uint32_t>(9));  // input tile-rows per image (its own H padding included)
        return;
    }

    if constexpr (input_resident) {
        // The shard's valid tile-rows, at the nominal (shard-width) block_width pages each.
        MaybeDeviceZoneScope("reader_publish_resident");
        const uint32_t pages = core_row_tiles * block_width;
        cb_reserve_back(cb_input_sticks, pages);
        cb_push_back(cb_input_sticks, pages);
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);

    if constexpr (coalesce_depth != 0) {
        read_bank_coalesced<
            cb_input_sticks,
            cb_retile_staging,  // the reader-private staging slot (retile and bank_coalesced are disjoint)
            block_width,
            tile_h,
            block_width * tile_col_bytes,
            stick_page_bytes,
            rows_per_quantum,
            coalesce_depth,
            NUM_DRAM_BANKS,
            coalesce_scatter_write>(
            input_accessor, row_start, core_row_tiles, col_start, core_col_tiles, row_rotation, stick_rotation);
        return;
    }

    tilize_dataflow::Walker<block_width> walk(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
    tilize_dataflow::StickProducer<
        cb_input_sticks,
        block_width,
        depth_in,
        read_ahead,
        tile_h,
        tile_col_bytes,
        rows_per_quantum,
        page_bytes,
        pages_per_stick,
        read_noc_split,
        eager_publish,
        bank_stride ? NUM_DRAM_BANKS : 0,
        bank_stride == 2,
        co_read,
        co_read_listed>
        producer(stick_rotation);
    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);
    if constexpr (co_read_listed) {
        // RT arg 10..: the co-read stick lists (co-read excludes the padded path, the only other
        // user of RT args >= 10)
        producer.co_read_list_arg = tilize_dataflow::select_co_read_list<tile_h>(10);
    }

    const uint32_t num_positions = walk.num_positions();
    if constexpr (padded) {
        // RT arg 10: the fill value packed per input dtype; 11..: the per-image stick map.
        const tilize_dataflow::PadMap<tile_h, tile_col_bytes> pad_map(11);
        tilize_dataflow::PadFill<elem_bytes, pad_source_bytes, pad_noc_min_bytes, depth_in + 1> fill(
            cb_pad_source, get_arg_val<uint32_t>(10));
        if (w_tail_persist && core_col_tiles <= block_width) {
            producer.set_w_tail_persists();  // one column block: every tile-row has the same W tail
        }
        for (uint32_t seq = 0; seq < num_positions; ++seq, walk.advance()) {
            const uint32_t first_col = walk.first_col();
            const uint32_t valid_width = walk.valid_width();
            producer.issue_row_padded(
                input_accessor, pad_map.source(walk.row(), first_col, valid_width), first_col, valid_width, fill);
        }
        producer.complete_all(fill);
        return;
    }
    for (uint32_t seq = 0; seq < num_positions; ++seq, walk.advance()) {
        if (!split_reader || (seq & 1) == 0) {
            producer.issue_row(input_accessor, walk.row(), walk.first_col(), walk.valid_width());
        }
    }
    if constexpr (co_read != 0) {
        // Co-read (host-gated to a one-position walk): the slot is published only once the
        // writer's share of its sticks has landed too.
        tilize_dataflow::CoReadLanded writer_landed(get_semaphore(co_read_sem));
        producer.complete_all(writer_landed);
    } else {
        producer.complete_all();
    }
}
