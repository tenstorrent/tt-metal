// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1) — the `store_block` block operation (op_design.md),
// plus, under the split reader, the odd half of `load_block`.
//
// No kernel_lib dataflow helper writes TILE pages (write_sticks_after_untilize
// writes ROW_MAJOR sticks), so this is a custom block operation.
//
// store_block, per CB quantum of rows_per_quantum tile-rows: wait the quantum's
// pages -> valid_width tile-page writes per tile-row in flight -> one flush (L1
// source reads done) -> pop the quantum. The flush, rather than a full write
// barrier, is enough to release the CB slots; one write barrier at kernel end
// guarantees the data has landed. Only the kernel's final quantum may be partial.
//
// Split reader (CT `split_reader`): this RISC-V also produces the ODD walk
// positions into cb_input_sticks_odd (its own CB, single producer). At odd
// position q it issues q's reads, and while they are in flight it stores every
// position whose inputs are already published (all positions <= the last odd
// position it pushed). Deadlock-free: each store waits on a position whose even
// input NCRISC produces independently and whose odd input this RISC-V has
// already pushed; the output CB (depth_out tile-rows) is drained in order.
//
// Tile-rows are walked with the same per-core rotation as the reader
// (tilize_stick_reads.hpp), so the CB FIFO order agrees.
//
// Co-read (CT `co_read`, Refinement 8): on a one-position walk this RISC-V first reads the last
// co_read sticks of the tile-row into cb_input_sticks' slot (NCRISC reads the rest and stays
// the CB's only producer), then raises the landed flag NCRISC waits on before its push. Which
// sticks is the host's geometric split (Perf 2: tilize_stick_reads.hpp select_co_read_list).
//
// Resident output (CT `output_resident`, sharded_resident regime):
// cb_output_tiles is backed on this Tensix core's own output shard and compute
// packs straight into it, so store_block issues no NoC write; this kernel only
// waits for the shard to be complete (it stays the CB's single consumer).
//
// Column sub-blocks (define TILIZE_SUB_BLOCK_TILES, Perf 2 onepos_pipeline; host knob
// SUB_BLOCK_TILES): compute pushes each tile-row as column sub-blocks in production order
// (tilize_sub_blocks.hpp), and store_row_sub_blocked writes each sub-block as soon as it is packed
// instead of waiting for the whole tile-row, in exactly store_rows' tile order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tilize_stick_reads.hpp"

#if defined(TILIZE_SUB_BLOCK_TILES) && defined(ARCH_WORMHOLE)
#include "tilize_sub_blocks.hpp"
namespace tilize_sub_blocks {

// store_rows (tilize_stick_reads.hpp) for ONE tile-row whose pages compute pushes as
// SubBlocks<block_width, sb_tiles> sub-blocks in production order. store_rows writes columns
// t0, t0 + 1, ..., wrapping (t0 = col_rotation mod valid_width); here that sequence is regrouped
// by sub-block: sub-block j0 (holding t0) is produced first and its columns [t0, end) written at
// once, the other sub-blocks follow as they land, and j0's columns [first, t0) go last. Pops the
// tile-row's block_width pages. Zones as store_rows' (writer_issue also covers the waits on the
// later sub-blocks, which interleave with the writes).
template <
    uint32_t cb,
    uint32_t block_width,
    uint32_t out_tile_bytes,
    uint32_t sb_tiles,
    typename Accessor,
    typename Walk>
FORCE_INLINE void store_row_sub_blocked(
    const Accessor& accessor, Walk& walk, uint32_t tiles_per_row, uint32_t col_rotation) {
    using SB = SubBlocks<block_width, sb_tiles>;
    const uint32_t row_tile_idx = walk.row() * tiles_per_row + walk.first_col();
    const uint32_t valid_width = walk.valid_width();
    const uint32_t base = get_read_ptr(cb);
    const uint32_t t0 = col_rotation % valid_width;
    const uint32_t j0 = SB::start(col_rotation, valid_width);
    auto write_cols = [&](uint32_t cb_off, uint32_t first, uint32_t lo, uint32_t hi) {
        for (uint32_t c = lo; c < hi; ++c) {
            noc_async_write(
                base + (cb_off + c - first) * out_tile_bytes, accessor.get_noc_addr(row_tile_idx + c), out_tile_bytes);
        }
    };
    {
        MaybeDeviceZoneScope("writer_wait");  // starved on read + the first sub-block's tilize
        cb_wait_front(cb, SB::width(j0));
    }
    {
        MaybeDeviceZoneScope("writer_issue");
        uint32_t cb_off = 0;  // CB page of the current sub-block's first tile (production order)
        for (uint32_t p = 0; p < SB::n; ++p) {
            const uint32_t k = SB::at(j0, p);
            const uint32_t first = SB::first(k);
            const uint32_t end = first + SB::width(k);
            if (p != 0) {
                cb_wait_front(cb, cb_off + SB::width(k));
            }
            write_cols(cb_off, first, p == 0 ? t0 : first, end < valid_width ? end : valid_width);
            cb_off += SB::width(k);
        }
        write_cols(0, SB::first(j0), SB::first(j0), t0);
    }
    {
        MaybeDeviceZoneScope("writer_flush");  // the tile-row's L1 source reads done (not landed)
        noc_async_writes_flushed();
    }
    cb_pop_front(cb, block_width);
    walk.advance();
}

}  // namespace tilize_sub_blocks
#endif  // TILIZE_SUB_BLOCK_TILES && ARCH_WORMHOLE

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);     // tiles per column block (CB quantum)
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);  // one output TILE page
    constexpr bool split_reader = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_input_sticks_odd = get_compile_time_arg_val(4);
    constexpr uint32_t tile_h = get_compile_time_arg_val(5);
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t depth_in = get_compile_time_arg_val(8);
    constexpr uint32_t read_ahead = get_compile_time_arg_val(9);
    constexpr uint32_t rows_per_quantum = get_compile_time_arg_val(10);  // tile-rows per CB quantum
    constexpr bool output_resident = get_compile_time_arg_val(11) != 0;  // cb_output_tiles backed on the shard
    constexpr uint32_t page_bytes = get_compile_time_arg_val(12);        // data bytes per input page
    constexpr uint32_t pages_per_stick = get_compile_time_arg_val(13);   // input pages per logical stick
    constexpr uint32_t write_noc_split = get_compile_time_arg_val(14);   // parked: 0, else other-NoC write period
    constexpr uint32_t depth_out = get_compile_time_arg_val(15);         // cb_output_tiles slots (quanta)
    constexpr uint32_t write_ahead = get_compile_time_arg_val(16);       // CB quanta of tile writes in flight
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(17);   // co-read: the slot this RISC-V fills
    constexpr uint32_t co_read = get_compile_time_arg_val(18);           // 0, else sticks per tile-row read here
    constexpr uint32_t co_read_sem = get_compile_time_arg_val(19);       // co-read: the landed flag's semaphore id
    constexpr auto output_args = TensorAccessorArgs<20>();
    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(5);  // C: output tile-columns of the whole tensor
    const uint32_t row_rotation = get_arg_val<uint32_t>(6);
    const uint32_t src_addr = get_arg_val<uint32_t>(7);  // input stick buffer (split reader only)
    const uint32_t stick_rotation = get_arg_val<uint32_t>(8);

    static_assert(co_read == 0 || !split_reader, "co-read and the split reader are exclusive");
    if constexpr (co_read != 0) {
        // Co-read (host-gated to a one-position walk, so the CB is empty and its first slot is
        // where NCRISC's reserve lands): read the last co_read sticks of the tile-row on this
        // RISC-V's NoC, then flag them landed. NCRISC stays cb_input_sticks' only producer.
        MaybeDeviceZoneScope("writer_coread");  // BRISC's share of the stick reads, issue + barrier
        tilize_dataflow::Walker<block_width> load_walk(
            row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
#ifdef CO_READ_LISTED
        // Perf 2: the host's geometric stick list for this RISC-V (RT arg 9..)
        tilize_dataflow::read_tile_row_sticks_listed<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            tilize_dataflow::select_co_read_list<tile_h>(9));
#else
        tilize_dataflow::read_tile_row_sticks<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            tile_h - co_read,
            tile_h);
#endif
        noc_async_read_barrier();
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(co_read_sem)), 1);
    }

    if constexpr (output_resident) {
        MaybeDeviceZoneScope("writer_wait");  // resident output: wait for compute to fill the shard
        cb_wait_front(cb_output_tiles, core_row_tiles * block_width);
        return;
    }

    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);

    tilize_dataflow::Walker<block_width> store_walk(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
    const uint32_t num_positions = store_walk.num_positions();

#if defined(TILIZE_SUB_BLOCK_TILES) && defined(ARCH_WORMHOLE)
    // Host-engaged (SUB_BLOCK_TILES) wherever this RISC-V streams the output with store_rows (no
    // split reader, one write quantum in flight, no write NoC split), on every walk; compute makes
    // the same decision from the same define.
    static_assert(
        !output_resident && !split_reader && write_ahead == 1 && write_noc_split == 0, "sub-blocks: store_rows path");
    for (uint32_t seq = 0; seq < num_positions; ++seq) {
        tilize_sub_blocks::store_row_sub_blocked<cb_output_tiles, block_width, out_tile_bytes, TILIZE_SUB_BLOCK_TILES>(
            output_accessor, store_walk, tiles_per_row, stick_rotation);
    }
    MaybeDeviceZoneScope("writer_barrier");  // every tile write landed
    noc_async_write_barrier();
    return;
#endif

    // Split reader runs one tile-row per quantum (host-enforced), so this stores one position.
    auto store_next = [&]() {
        tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes>(
            output_accessor, store_walk, tiles_per_row, 1, stick_rotation);
    };

    static_assert(!split_reader || rows_per_quantum == 1, "the split reader alternates CBs per tile-row");
    if constexpr (split_reader) {
        const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);
        tilize_dataflow::Walker<block_width> load_walk(
            row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
        tilize_dataflow::StickProducer<
            cb_input_sticks_odd,
            block_width,
            depth_in,
            read_ahead,
            tile_h,
            tile_col_bytes,
            1,
            page_bytes,
            pages_per_stick>
            producer(stick_rotation);

        uint32_t stored = 0;     // positions [0, stored) are written
        uint32_t published = 0;  // positions [0, published) have all their inputs pushed
        for (uint32_t seq = 0; seq < num_positions; ++seq, load_walk.advance()) {
            if ((seq & 1) == 0) {
                continue;
            }
            const uint32_t pushed_before = producer.rows_pushed();  // odd items pushed
            producer.issue_row(input_accessor, load_walk.row(), load_walk.first_col(), load_walk.valid_width());
            const uint32_t pushed_after = producer.rows_pushed();
            if (pushed_after != pushed_before) {
                // The k-th odd item (k = pushed_after - 1) is position 2k + 1: all positions <= it are publishable.
                published = 2 * pushed_after;
            }
            while (stored < published) {
                store_next();
                ++stored;
            }
        }
        producer.complete_all();
        while (stored < num_positions) {
            store_next();
            ++stored;
        }
    } else if constexpr (write_ahead > 1) {
        tilize_dataflow::
            TileStorer<cb_output_tiles, block_width, out_tile_bytes, depth_out, write_ahead, rows_per_quantum>
                storer;
        for (uint32_t done = 0; done < num_positions; done += rows_per_quantum) {
            const uint32_t remaining = num_positions - done;
            storer.store(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum,
                stick_rotation);
        }
        storer.complete_all();
    } else {
        for (uint32_t done = 0; done < num_positions; done += rows_per_quantum) {
            const uint32_t remaining = num_positions - done;
            tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes, write_noc_split>(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum,
                stick_rotation);
        }
    }
    MaybeDeviceZoneScope("writer_barrier");  // every tile write landed
    noc_async_write_barrier();
    if constexpr (write_noc_split != 0) {
        noc_async_write_barrier(1 - noc_index);
    }
}
