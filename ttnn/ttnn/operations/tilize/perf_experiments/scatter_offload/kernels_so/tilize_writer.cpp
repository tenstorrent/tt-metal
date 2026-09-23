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
// the CB's only producer), then raises the landed flag NCRISC waits on before its push.
//
// Resident output (CT `output_resident`, sharded_resident regime):
// cb_output_tiles is backed on this Tensix core's own output shard and compute
// packs straight into it, so store_block issues no NoC write; this kernel only
// waits for the shard to be complete (it stays the CB's single consumer).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tilize_stick_reads.hpp"
#include "so_coalesced.hpp"

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
        tilize_dataflow::read_tile_row_sticks<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            tile_h - co_read,
            tile_h);
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

#if SO_MODE != 0
    {
        // scatter_offload (so_coalesced.hpp): BRISC's share of the bank_coalesced scatter (and,
        // mode 3, of its DRAM reads), interleaved with store_block by polling -- never a blocking
        // CB wait, so neither the scatter wait nor the output drain can starve the other.
        constexpr uint32_t stage_depth = SO_STAGE;
        constexpr uint32_t stick_bytes = block_width * tile_col_bytes;
        constexpr uint32_t scatter_trid = stage_depth + 1;
        constexpr uint32_t stage_slot_bytes = rows_per_quantum * tile_h * stick_page_bytes;
        constexpr uint32_t in_slot_bytes = rows_per_quantum * tile_h * stick_bytes;
        using ReadShare =
            so::Share<block_width, tile_h, stick_bytes, stick_page_bytes, NUM_DRAM_BANKS, so::nc_num, so::den>;
        using ScatterShare = ReadShare;  // BRISC scatters ordinals [nc_num, den) in every mode
        volatile tt_l1_ptr uint32_t* staged = so::sem_ptr(SO_SEM_STAGED);
        volatile tt_l1_ptr uint32_t* done = so::sem_ptr(SO_SEM_DONE);
        // Mode 3: BRISC's own staging ring (the host doubled the staging CB), after NCRISC's.
        const uint32_t stage_base = get_write_ptr(SO_CB_STAGING) + (SO_MODE == 3 ? stage_depth * stage_slot_bytes : 0);
        const uint32_t in_base = get_write_ptr(cb_input_sticks);
        const uint32_t num_units = (num_positions + rows_per_quantum - 1) / rows_per_quantum;
        auto unit_rows = [&](uint32_t k) {
            const uint32_t left = num_positions - k * rows_per_quantum;
            return left < rows_per_quantum ? left : rows_per_quantum;
        };
        tilize_dataflow::Walker<block_width> scatter_walk(
            row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
#if SO_MODE == 3
        const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);
        tilize_dataflow::Walker<block_width> issue_walk(
            row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
        uint32_t issued = 0;
        auto issue = [&]() {
            MaybeDeviceZoneScope("so_brisc_issue");
            noc_async_read_set_trid(1 + (issued % stage_depth));
            ReadShare::issue(
                input_accessor,
                issue_walk,
                unit_rows(issued),
                stage_base + (issued % stage_depth) * stage_slot_bytes,
                stick_rotation);
            ++issued;
        };
        while (issued < num_units && issued < stage_depth) {
            issue();
        }
#endif
        uint32_t scattered = 0, stored = 0;
        while (stored < num_units) {
#ifdef SO_PRIO_STORE
            // Store priority: the scatter share only fills BRISC's idle time.
            if (stored < scattered || scattered == num_units) {
                const uint32_t rows = unit_rows(stored);
                if (cb_pages_available_at_front(cb_output_tiles, rows * block_width)) {
                    tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes>(
                        output_accessor, store_walk, tiles_per_row, rows, stick_rotation);
                    ++stored;
                    continue;
                }
            }
#endif
            if (scattered < num_units && so::sem_read(staged) > scattered) {
                const uint32_t u = scattered;
#if SO_MODE == 3
                {
                    MaybeDeviceZoneScope("so_brisc_barrier");
                    noc_async_read_barrier_with_trid(1 + (u % stage_depth));
                }
#endif
                {
                    MaybeDeviceZoneScope("so_brisc_scatter");
                    noc_async_read_set_trid(scatter_trid);
                    noc_async_read_one_packet_set_state(get_noc_addr(0), stick_bytes);
                    ScatterShare::scatter(
                        scatter_walk,
                        unit_rows(u),
                        stage_base + (u % stage_depth) * stage_slot_bytes,
                        in_base + (u % depth_in) * in_slot_bytes,
                        stick_rotation);
                    noc_async_read_barrier_with_trid(scatter_trid);
                }
                asm volatile("" ::: "memory");
                *done = u + 1;
                ++scattered;
#if SO_MODE == 3
                if (issued < num_units) {
                    issue();  // lands in the staging slot unit u was just scattered out of
                }
#endif
                continue;
            }
            const uint32_t rows = unit_rows(stored);
            if (cb_pages_available_at_front(cb_output_tiles, rows * block_width)) {
                tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes>(
                    output_accessor, store_walk, tiles_per_row, rows, stick_rotation);
                ++stored;
            }
        }
        noc_async_read_set_trid(0);
        MaybeDeviceZoneScope("writer_barrier");
        noc_async_write_barrier();
        return;
    }
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
