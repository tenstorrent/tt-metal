// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1) — block operation `store_tiles_block`.
//
// Mirrors the reader's block walk. Per tile-row: wait `width` tiles, issue `width` tile-page
// writes, flush (L1 slot reusable), pop `width`; one write barrier at the end. out_tile_id = r * tensor_cols_tiles + c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles_out_tail = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(6);
    constexpr bool has_tail = get_compile_time_arg_val(7) != 0;
    // Resident output (sharded regimes): cb_tiles_out is backed on this core's own output shard, so
    // the compute packs straight into place; the writer only retires the pages.
    constexpr bool out_resident = get_compile_time_arg_val(8) != 0;
    // Split reorder (coalesced reader only; one column block): this RISC also copies the last
    // writer_copy_sticks sticks of every tile-row from the reader's staging slot into cb_rm_sticks,
    // as self-aimed reads on its own NoC (it issues no other reads, so a plain read barrier
    // retires exactly the copies). Job = {dst, staging slot base, first stick in the chunk}.
    constexpr uint32_t writer_copy_sticks_arg = get_compile_time_arg_val(9);  // 0 = reader copies all
    constexpr uint32_t cb_copy_job = get_compile_time_arg_val(10);
    constexpr uint32_t cb_copy_done = get_compile_time_arg_val(11);
    constexpr uint32_t copy_seg_bytes = get_compile_time_arg_val(12);  // bytes of one stick copy
    constexpr uint32_t staging_num_banks = get_compile_time_arg_val(13);
    constexpr uint32_t staging_group_bytes = get_compile_time_arg_val(14);  // one bank group in a slot
    constexpr uint32_t staging_stick_bytes = get_compile_time_arg_val(15);
    constexpr auto out_args = TensorAccessorArgs<16>();
    // Same gate as the reader: the split applies to single-packet stick copies only.
    constexpr uint32_t writer_copy_sticks = copy_seg_bytes <= NOC_MAX_BURST_SIZE ? writer_copy_sticks_arg : 0;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(4);

    if constexpr (out_resident) {
        const uint32_t n_pages = core_rows_tiles * block_width_tiles;
        cb_wait_front(cb_tiles_out, n_pages);
        cb_pop_front(cb_tiles_out, n_pages);
        return;
    }

    const auto out_acc = TensorAccessor(out_args, dst_addr, out_page_bytes);

    if constexpr (writer_copy_sticks > 0) {
        static_assert(
            writer_copy_sticks == 0 || (num_col_blocks == 1 && !has_tail),
            "split reorder needs one whole-width column block");
        noc_async_read_one_packet_set_state(get_noc_addr(get_write_ptr(cb_copy_job)), copy_seg_bytes);
        uint32_t jobs_done = 0;
        auto do_copy_job = [&]() {
            cb_wait_front(cb_copy_job, 1);
            volatile tt_l1_ptr uint32_t* job =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_copy_job));
            uint32_t dst_l1 = job[0];
            const uint32_t base = job[1];
            const uint32_t s0 = job[2];
            cb_pop_front(cb_copy_job, 1);
            for (uint32_t s = s0; s < s0 + writer_copy_sticks; ++s) {
                const uint32_t src_l1 = base + (s % staging_num_banks) * staging_group_bytes +
                                        (s / staging_num_banks) * staging_stick_bytes;
                noc_async_read_one_packet_with_state(src_l1, dst_l1);
                dst_l1 += copy_seg_bytes;
            }
            noc_async_read_barrier();
            cb_reserve_back(cb_copy_done, 1);
            cb_push_back(cb_copy_done, 1);
            ++jobs_done;
        };
        for (uint32_t r = 0; r < core_rows_tiles; ++r) {
            // Tile-row r cannot reach this CB before its own copy job is done; while waiting for it,
            // keep serving the reader's later jobs so neither RISC idles on the other.
            while (jobs_done <= r) {
                do_copy_job();
            }
            while (true) {
                invalidate_l1_cache();  // polled CB counters: no stale cached value
                if (cb_pages_available_at_front(cb_tiles_out, block_width_tiles)) {
                    break;
                }
                if (jobs_done < core_rows_tiles && cb_pages_available_at_front(cb_copy_job, 1)) {
                    do_copy_job();
                }
            }
            cb_wait_front(cb_tiles_out, block_width_tiles);
            const uint32_t first_tile = (row_start_tile + r) * tensor_cols_tiles + col_start_tile;
            uint32_t l1_read_addr = get_read_ptr(cb_tiles_out);
            for (uint32_t c = 0; c < block_width_tiles; ++c) {
                noc_async_write(l1_read_addr, out_acc.get_noc_addr(first_tile + c), out_page_bytes);
                l1_read_addr += out_page_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_tiles_out, block_width_tiles);
        }
        noc_async_write_barrier();
        return;
    }

    for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
        const bool is_tail = has_tail && (col_block == num_col_blocks - 1);
        const uint32_t cb_id = is_tail ? cb_tiles_out_tail : cb_tiles_out;
        const uint32_t width_tiles = is_tail ? last_block_width_tiles : block_width_tiles;
        const uint32_t block_col_start = col_start_tile + col_block * block_width_tiles;

        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rb_end =
                (rb_start + block_height_tiles < core_rows_tiles) ? rb_start + block_height_tiles : core_rows_tiles;
            for (uint32_t r = rb_start; r < rb_end; ++r) {
                const uint32_t first_tile = (row_start_tile + r) * tensor_cols_tiles + block_col_start;
                cb_wait_front(cb_id, width_tiles);
                uint32_t l1_read_addr = get_read_ptr(cb_id);
                for (uint32_t c = 0; c < width_tiles; ++c) {
                    noc_async_write(l1_read_addr, out_acc.get_noc_addr(first_tile + c), out_page_bytes);
                    l1_read_addr += out_page_bytes;
                }
                // Flushed = data has left L1, so the slot may be recycled; acks are collected once
                // at the end, keeping writes of consecutive tile-rows in flight together.
                noc_async_writes_flushed();
                cb_pop_front(cb_id, width_tiles);
            }
        }
    }
    noc_async_write_barrier();
}
