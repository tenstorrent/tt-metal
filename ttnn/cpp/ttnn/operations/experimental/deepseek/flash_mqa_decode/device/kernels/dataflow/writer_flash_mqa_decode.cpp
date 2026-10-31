// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_common.hpp"

void kernel_main() {
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t num_tree_steps = get_named_compile_time_arg_val("num_tree_steps");
    constexpr uint32_t heads_per_core = get_named_compile_time_arg_val("heads_per_core");
    constexpr uint32_t out_row_bytes = get_named_compile_time_arg_val("out_row_bytes");
    constexpr uint32_t ring_capacity = get_named_compile_time_arg_val("ring_capacity");
    constexpr uint32_t selected_capacity = get_named_compile_time_arg_val("selected_capacity");
    constexpr bool has_sink = get_named_compile_time_arg_val("has_sink") == 1;
    constexpr uint32_t reducer_semaphore_id = get_named_compile_time_arg_val("reducer_semaphore_id");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_out_o = get_named_compile_time_arg_val("cb_out_o");
    constexpr uint32_t cb_out_ms = get_named_compile_time_arg_val("cb_out_ms");
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("cb_out_rm");
    constexpr uint32_t cb_lengths_scratch = get_named_compile_time_arg_val("cb_lengths_scratch");
    constexpr uint32_t cb_sink_scratch = get_named_compile_time_arg_val("cb_sink_scratch");

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto lengths_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<lengths_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t lengths_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sink_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t s_block = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_group = get_arg_val<uint32_t>(arg_idx++);
    // NoC (x, y) of the S-blocks of this head group, in S-block order.
    const uint32_t block_noc_xy_idx = arg_idx;

    const auto lengths_reader = TensorAccessor(lengths_args, lengths_addr);
    const uint32_t lengths_l1 = get_write_ptr(cb_lengths_scratch);
    noc_async_read(lengths_reader.get_noc_addr(0), lengths_l1, 2 * sizeof(uint32_t));
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* lengths_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lengths_l1);
    const flash_mqa_decode::Lengths lengths =
        flash_mqa_decode::clamp_lengths(lengths_ptr[0], lengths_ptr[1], ring_capacity, selected_capacity);
    const uint32_t total_chunks = flash_mqa_decode::num_chunks(lengths, k_chunk_size);
    if (s_block >= total_chunks) {
        return;
    }
    const uint32_t num_active = total_chunks < num_s_blocks ? total_chunks : num_s_blocks;
    const uint32_t num_received = flash_mqa_decode::num_received(s_block, num_active, num_tree_steps);

    constexpr uint32_t out_chunk_tiles = vDHt;
    const uint32_t ms_tile_bytes = get_tile_size(cb_ms_in);
    const uint32_t out_step_bytes = out_chunk_tiles * get_tile_size(cb_out_in);
    // Receive slots: one per tree step, then one for the sink, at the same L1 address on every core.
    const uint32_t ms_in_base = get_write_ptr(cb_ms_in);
    const uint32_t out_in_base = get_write_ptr(cb_out_in);

    if (s_block == 0) {
        // The sink joins the reduction as one more peer: max = sink / scale, sum = 1, O = 0. Without a
        // sink the peer is neutral (max = -1e30, sum = 0). Prepared now and pushed after the tree slots.
        const uint32_t sink_ms_l1 = ms_in_base + num_received * ms_tile_bytes;
        const uint32_t sink_out_l1 = out_in_base + num_received * out_step_bytes;
        const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        noc_async_read(zeros_noc_addr, sink_ms_l1, ms_tile_bytes);
        for (uint32_t offset = 0; offset < out_step_bytes; offset += MEM_ZEROS_SIZE) {
            noc_async_read(zeros_noc_addr, sink_out_l1 + offset, MEM_ZEROS_SIZE);
        }
        if constexpr (has_sink) {
            // The sink tile is TILE [heads, 32] bf16 with sink / scale in column 0.
            const auto sink_reader = TensorAccessor(sink_args, sink_addr);
            const uint32_t sink_l1 = get_write_ptr(cb_sink_scratch);
            const uint32_t first_head = head_group * heads_per_core;
            noc_async_read_page(first_head / 32, sink_reader, sink_l1);
        }
        noc_async_read_barrier();
        // ms tile is (heads_per_core, 32) in two faces of 16 columns: (h, c) sits at h * 16 + c.
        volatile tt_l1_ptr uint16_t* ms_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_ms_l1);
        if constexpr (has_sink) {
            volatile tt_l1_ptr uint16_t* sink_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_sink_scratch));
            const uint32_t first_row = (head_group * heads_per_core) % 32;
            for (uint32_t h = 0; h < heads_per_core; ++h) {
                const uint32_t row = first_row + h;
                // 32x32 tile: rows 16..31 live in faces 2 and 3; column 0 is in the even face.
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

    const uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);
    volatile tt_l1_ptr uint32_t* reducer_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    for (uint32_t step = 0; step < num_received; ++step) {
        cb_reserve_back(cb_ms_in, 1);
        cb_reserve_back(cb_out_in, out_chunk_tiles);
        const uint32_t step_bit = 1u << step;
        uint32_t sem_val;
        do {
            invalidate_l1_cache();
            sem_val = *reducer_semaphore_ptr;
        } while ((sem_val & step_bit) == 0);
        cb_push_back(cb_ms_in, 1);
        cb_push_back(cb_out_in, out_chunk_tiles);
    }
    if (num_received > 0) {
        noc_semaphore_set(reducer_semaphore_ptr, 0);
    }

    if (s_block != 0) {
        const uint32_t step = flash_mqa_decode::send_step(s_block);
        const uint32_t parent = s_block - (1u << step);
        const uint32_t parent_x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * parent);
        const uint32_t parent_y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * parent + 1);
        cb_wait_front(cb_out_ms, 1);
        noc_async_write(
            get_read_ptr(cb_out_ms),
            get_noc_addr(parent_x, parent_y, ms_in_base + step * ms_tile_bytes),
            ms_tile_bytes);
        cb_wait_front(cb_out_o, out_chunk_tiles);
        noc_async_write(
            get_read_ptr(cb_out_o),
            get_noc_addr(parent_x, parent_y, out_in_base + step * out_step_bytes),
            out_step_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(parent_x, parent_y, reducer_semaphore_addr), 1u << step);
        noc_async_atomic_barrier();
        cb_pop_front(cb_out_ms, 1);
        cb_pop_front(cb_out_o, out_chunk_tiles);
        return;
    }

    cb_reserve_back(cb_ms_in, 1);
    cb_reserve_back(cb_out_in, out_chunk_tiles);
    cb_push_back(cb_ms_in, 1);
    cb_push_back(cb_out_in, out_chunk_tiles);

    const auto out_writer = TensorAccessor(out_args, out_addr);
    cb_wait_front(cb_out_rm, out_chunk_tiles);
    uint32_t out_l1 = get_read_ptr(cb_out_rm);
    for (uint32_t h = 0; h < heads_per_core; ++h) {
        noc_async_write(out_l1, out_writer.get_noc_addr(head_group * heads_per_core + h), out_row_bytes);
        out_l1 += out_row_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_rm, out_chunk_tiles);
}
