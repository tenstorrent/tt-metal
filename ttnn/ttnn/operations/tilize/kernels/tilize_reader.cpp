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

void kernel_main() {
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
    constexpr uint32_t bank_stride = get_compile_time_arg_val(19);  // parked: 0 off, 1 bank-stride, 2 bank-major
    constexpr bool padded = get_compile_time_arg_val(20) != 0;      // output padded shape > input: fill
    constexpr uint32_t cb_pad_source = get_compile_time_arg_val(21);  // padded: reader-private fill source
    constexpr uint32_t pad_source_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t pad_noc_min_bytes = get_compile_time_arg_val(23);  // shorter fills are CPU stores
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(24);         // input element size
    constexpr auto input_args = TensorAccessorArgs<25>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t row_rotation = get_arg_val<uint32_t>(5);
    const uint32_t stick_rotation = get_arg_val<uint32_t>(6);

    if constexpr (in_tile_h != 0) {
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
        const uint32_t pages = core_row_tiles * block_width;
        cb_reserve_back(cb_input_sticks, pages);
        cb_push_back(cb_input_sticks, pages);
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);

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
        bank_stride == 2>
        producer(stick_rotation);
    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);

    const uint32_t num_positions = walk.num_positions();
    if constexpr (padded) {
        // RT arg 10: the fill value packed per input dtype; 11..: the per-image stick map.
        static_assert(!split_reader, "the split reader has no pad path");
        static_assert(depth_in + 1 <= 15, "the fill's transaction id follows the slots' 1..depth_in");
        const tilize_dataflow::PadMap<tile_h, tile_col_bytes> pad_map(11);
        tilize_dataflow::PadFill<elem_bytes, pad_source_bytes, pad_noc_min_bytes, depth_in + 1> fill(
            cb_pad_source, get_arg_val<uint32_t>(10));
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
    producer.complete_all();
}
