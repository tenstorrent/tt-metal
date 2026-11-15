// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-mode writer (num_rows > 1). Per row j of this core's row group, the tree reduction of the one-row writer over
// row j's active S-blocks of the group, with row j's receive slots after the group's earlier rows' ones (the CB never
// wraps, so a sender computes its parent's slot address) and one semaphore per group row. The output is tiled by the
// row groups' roots (S-block 0) together: root rg owns heads [rg * hpr, rg * hpr + hpr) (hpr = 8 / row_groups) of
// this head group. Each root sends row j's [head_dim] row of every head to the root owning the head, at row
// h_local * R + j of its tilize input, and counts it on that root's gather semaphore; each root then tiles its
// [hpr * R, head_dim] block into (R, 32) tiles and writes them.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_rows_common.hpp"

void kernel_main() {
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t num_tree_steps = get_named_compile_time_arg_val("num_tree_steps");
    constexpr uint32_t heads_per_core = get_named_compile_time_arg_val("heads_per_core");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t row_groups = get_named_compile_time_arg_val("row_groups");
    constexpr uint32_t group_rows = num_rows / row_groups;
    constexpr bool dense = get_named_compile_time_arg_val("dense_selected") == 1;
    constexpr uint32_t out_row_bytes = get_named_compile_time_arg_val("out_row_bytes");
    constexpr uint32_t ring_capacity = get_named_compile_time_arg_val("ring_capacity");
    constexpr uint32_t selected_capacity = get_named_compile_time_arg_val("selected_capacity");
    constexpr bool has_sink = get_named_compile_time_arg_val("has_sink") == 1;
    constexpr uint32_t first_semaphore_id = get_named_compile_time_arg_val("first_semaphore_id");
    constexpr uint32_t out_run_pages = get_named_compile_time_arg_val("out_run_pages");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_out_o = get_named_compile_time_arg_val("cb_out_o");
    constexpr uint32_t cb_out_ms = get_named_compile_time_arg_val("cb_out_ms");
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("cb_out_rm");
    constexpr uint32_t cb_lengths_scratch = get_named_compile_time_arg_val("cb_lengths_scratch");
    constexpr uint32_t cb_sink_scratch = get_named_compile_time_arg_val("cb_sink_scratch");
    constexpr uint32_t cb_tile_rm = get_named_compile_time_arg_val("cb_tile_rm");
    constexpr uint32_t cb_out_tiles = get_named_compile_time_arg_val("cb_out_tiles");

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto lengths_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<lengths_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t lengths_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sink_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t s_block = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_group = get_arg_val<uint32_t>(arg_idx++);
    // NoC (x, y) of the row groups' roots of this head group, in row-group order.
    const uint32_t root_noc_xy_idx = arg_idx;
    arg_idx += 2 * row_groups;
    constexpr uint32_t heads_per_root = heads_per_core / row_groups;
    // NoC (x, y) of the S-blocks of this head group and row group, in S-block order.
    const uint32_t block_noc_xy_idx = arg_idx;

    const auto lengths_reader = TensorAccessor(lengths_args, lengths_addr);
    const uint32_t lengths_l1 = get_write_ptr(cb_lengths_scratch);
    noc_async_read(lengths_reader.get_noc_addr(0), lengths_l1, (4 + num_rows) * sizeof(uint32_t));
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* lengths_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lengths_l1);
    uint32_t n[num_rows];
    for (uint32_t j = 0; j < num_rows; ++j) {
        n[j] = lengths_ptr[4 + j];
    }
    const flash_mqa_decode_rows::Plan plan = flash_mqa_decode_rows::make_plan(
        num_rows, ring_capacity, k_chunk_size, selected_capacity, dense, lengths_ptr[0], n, row_group, row_groups);

    bool any_active = false;
    for (uint32_t j = row_group; j < num_rows; j += row_groups) {
        any_active |= s_block < flash_mqa_decode_rows::row_active(plan, j, num_s_blocks);
    }
    if (!any_active) {
        return;
    }

    constexpr uint32_t out_chunk_tiles = vDHt;
    const uint32_t ms_tile_bytes = get_tile_size(cb_ms_in);
    const uint32_t out_step_bytes = out_chunk_tiles * get_tile_size(cb_out_in);
    const uint32_t ms_in_base = get_write_ptr(cb_ms_in);
    const uint32_t out_in_base = get_write_ptr(cb_out_in);
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    if (s_block == 0) {
        if constexpr (has_sink) {
            const auto sink_reader = TensorAccessor(sink_args, sink_addr);
            noc_async_read_page((head_group * heads_per_core) / 32, sink_reader, get_write_ptr(cb_sink_scratch));
        }
    }
    if (s_block == 0) {
        // Row h_local * R + j of the tilize input; the whole block is pushed once every row is in.
        cb_reserve_back(cb_tile_rm, heads_per_root * vDHt);
    }
    const uint32_t tile_rm_base = get_write_ptr(cb_tile_rm);
    const uint32_t gather_sem_addr = get_semaphore(first_semaphore_id + group_rows);

    uint32_t slot = 0;  // this core's receive slots used by earlier rows
    for (uint32_t j = row_group, jl = 0; j < num_rows; j += row_groups, ++jl) {
        const uint32_t active = flash_mqa_decode_rows::row_active(plan, j, num_s_blocks);
        if (s_block >= active) {
            continue;
        }
        const uint32_t num_received = flash_mqa_decode::num_received(s_block, active, num_tree_steps);
        const uint32_t sem_addr = get_semaphore(first_semaphore_id + jl);
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

        if (s_block == 0) {
            // The sink peer: max = sink / scale, sum = 1, O = 0 (neutral max = -1e30, sum = 0 without a sink).
            const uint32_t sink_ms_l1 = ms_in_base + (slot + num_received) * ms_tile_bytes;
            const uint32_t sink_out_l1 = out_in_base + (slot + num_received) * out_step_bytes;
            noc_async_read(zeros_noc_addr, sink_ms_l1, ms_tile_bytes);
            for (uint32_t offset = 0; offset < out_step_bytes; offset += MEM_ZEROS_SIZE) {
                noc_async_read(zeros_noc_addr, sink_out_l1 + offset, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* ms_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_ms_l1);
            if constexpr (has_sink) {
                volatile tt_l1_ptr uint16_t* sink_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_sink_scratch));
                const uint32_t first_row = (head_group * heads_per_core) % 32;
                for (uint32_t h = 0; h < heads_per_core; ++h) {
                    const uint32_t row = first_row + h;
                    const uint32_t sink_idx = (row / 16) * 512 + (row % 16) * 16;
                    ms_ptr[h * 16 + 0] = sink_ptr[sink_idx];
                    ms_ptr[h * 16 + 1] = 0x3F80;
                }
            } else {
                for (uint32_t h = 0; h < heads_per_core; ++h) {
                    ms_ptr[h * 16 + 0] = 0xF14A;
                }
            }
        }

        for (uint32_t step = 0; step < num_received; ++step) {
            cb_reserve_back(cb_ms_in, 1);
            cb_reserve_back(cb_out_in, out_chunk_tiles);
            const uint32_t step_bit = 1u << step;
            uint32_t sem_val;
            do {
                invalidate_l1_cache();
                sem_val = *sem_ptr;
            } while ((sem_val & step_bit) == 0);
            cb_push_back(cb_ms_in, 1);
            cb_push_back(cb_out_in, out_chunk_tiles);
        }
        if (num_received > 0) {
            noc_semaphore_set(sem_ptr, 0);
        }

        if (s_block != 0) {
            const uint32_t step = flash_mqa_decode::send_step(s_block);
            const uint32_t parent = s_block - (1u << step);
            uint32_t parent_slot = step;
            for (uint32_t jj = row_group; jj < j; jj += row_groups) {
                parent_slot += flash_mqa_decode_rows::receive_pages(plan, parent, jj, num_s_blocks, num_tree_steps);
            }
            const uint32_t parent_x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * parent);
            const uint32_t parent_y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * parent + 1);
            cb_wait_front(cb_out_ms, 1);
            noc_async_write(
                get_read_ptr(cb_out_ms),
                get_noc_addr(parent_x, parent_y, ms_in_base + parent_slot * ms_tile_bytes),
                ms_tile_bytes);
            cb_wait_front(cb_out_o, out_chunk_tiles);
            noc_async_write(
                get_read_ptr(cb_out_o),
                get_noc_addr(parent_x, parent_y, out_in_base + parent_slot * out_step_bytes),
                out_step_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(parent_x, parent_y, sem_addr), 1u << step);
            noc_async_atomic_barrier();
            cb_pop_front(cb_out_ms, 1);
            cb_pop_front(cb_out_o, out_chunk_tiles);
        } else {
            cb_reserve_back(cb_ms_in, 1);
            cb_reserve_back(cb_out_in, out_chunk_tiles);
            cb_push_back(cb_ms_in, 1);
            cb_push_back(cb_out_in, out_chunk_tiles);

            // Head h's row j goes to row (h % hpr) * R + j of the tilize input of root h / hpr.
            cb_wait_front(cb_out_rm, out_chunk_tiles);
            uint32_t src = get_read_ptr(cb_out_rm);
            for (uint32_t r = 0; r < row_groups; ++r) {
                const uint32_t x = get_arg_val<uint32_t>(root_noc_xy_idx + 2 * r);
                const uint32_t y = get_arg_val<uint32_t>(root_noc_xy_idx + 2 * r + 1);
                for (uint32_t hl = 0; hl < heads_per_root; ++hl) {
                    noc_async_write(
                        src, get_noc_addr(x, y, tile_rm_base + (hl * num_rows + j) * out_row_bytes), out_row_bytes);
                    src += out_row_bytes;
                }
            }
            noc_async_write_barrier();
            for (uint32_t r = 0; r < row_groups; ++r) {
                const uint32_t x = get_arg_val<uint32_t>(root_noc_xy_idx + 2 * r);
                const uint32_t y = get_arg_val<uint32_t>(root_noc_xy_idx + 2 * r + 1);
                noc_semaphore_inc(get_noc_addr(x, y, gather_sem_addr), 1);
            }
            noc_async_atomic_barrier();
            cb_pop_front(cb_out_rm, out_chunk_tiles);
        }
        slot += num_received + (s_block == 0 ? 1 : 0);
    }

    if (s_block != 0) {
        return;
    }
    {
        volatile tt_l1_ptr uint32_t* gather_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_sem_addr);
        noc_semaphore_wait(gather_sem, num_rows);
        noc_semaphore_set(gather_sem, 0);
    }
    cb_push_back(cb_tile_rm, heads_per_root * vDHt);

    // (R, 32) tiles of this root's heads: pages [first, first + hpr * vDHt), written in runs of pages that are
    // contiguous in the output buffer.
    constexpr uint32_t num_pages = heads_per_root * vDHt;
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const uint32_t page_bytes = get_tile_size(cb_out_tiles);
    const uint32_t first_page = (head_group * heads_per_core + row_group * heads_per_root) * vDHt;
    cb_wait_front(cb_out_tiles, num_pages);
    uint32_t l1 = get_read_ptr(cb_out_tiles);
    uint32_t page = first_page;
    while (page < first_page + num_pages) {
        uint32_t run_end = (page / out_run_pages + 1) * out_run_pages;
        run_end = run_end < first_page + num_pages ? run_end : first_page + num_pages;
        const uint32_t bytes = (run_end - page) * page_bytes;
        noc_async_write(l1, out_writer.get_noc_addr(page), bytes);
        l1 += bytes;
        page = run_end;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_tiles, num_pages);
}
