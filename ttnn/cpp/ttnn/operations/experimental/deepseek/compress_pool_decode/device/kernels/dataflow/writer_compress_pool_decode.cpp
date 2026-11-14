// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace {

inline void copy_l1_words(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    for (uint32_t i = 0; i < bytes / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

inline void zero_l1_async(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        const uint32_t len = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, len);
        addr += len;
        bytes -= len;
    }
}

inline void extract_tile_row(uint32_t dst_row_addr, uint32_t src_tile_addr, uint32_t row) {
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;
    const uint32_t row_offset = (row / 16) * 2 * face_bytes + (row % 16) * face_row_bytes;
    copy_l1_words(dst_row_addr, src_tile_addr + row_offset, face_row_bytes);
    copy_l1_words(dst_row_addr + face_row_bytes, src_tile_addr + row_offset + face_bytes, face_row_bytes);
}

inline void patch_tile_row(uint32_t dst_tile_addr, uint32_t src_row_addr, uint32_t row) {
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;
    const uint32_t row_offset = (row / 16) * 2 * face_bytes + (row % 16) * face_row_bytes;
    copy_l1_words(dst_tile_addr + row_offset, src_row_addr, face_row_bytes);
    copy_l1_words(dst_tile_addr + row_offset + face_bytes, src_row_addr + face_row_bytes, face_row_bytes);
}

// The 1x32 RMS stats of cores [first, first + num_stats) of the common-arg core list (slot: this core's
// place among them) gather on the first (at hub_x, hub_y), whose compute turns them into a scale; the
// scale then reaches all of them in scale_cb.
template <uint32_t num_stats, uint32_t row_tile_bytes>
inline void gather_rms_scale(
    uint32_t slot,
    uint32_t first,
    uint32_t hub_x,
    uint32_t hub_y,
    uint32_t local_stat_cb_id,
    uint32_t gather_cb_id,
    uint32_t scale_hub_cb_id,
    uint32_t scale_cb_id,
    uint32_t arrive_sem_id,
    uint32_t ready_sem_id) {
    CircularBuffer local_stat_cb(local_stat_cb_id);
    CircularBuffer gather_cb(gather_cb_id);
    CircularBuffer scale_cb(scale_cb_id);

    const uint32_t arrive_sem_addr = get_semaphore(arrive_sem_id);
    volatile tt_l1_ptr uint32_t* arrive_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem_addr);
    const uint32_t ready_sem_addr = get_semaphore(ready_sem_id);
    volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);

    if (slot == 0) {
        gather_cb.reserve_back(num_stats);
        const uint32_t gather_l1 = gather_cb.get_write_ptr();
        local_stat_cb.wait_front(1);
        invalidate_l1_cache();
        copy_l1_words(gather_l1, local_stat_cb.get_read_ptr(), row_tile_bytes);
        local_stat_cb.pop_front(1);
        if constexpr (num_stats > 1) {
            noc_semaphore_wait(arrive_sem, num_stats - 1);
            noc_semaphore_set(arrive_sem, 0);
        }
        gather_cb.push_back(num_stats);

        CircularBuffer scale_hub_cb(scale_hub_cb_id);
        scale_hub_cb.wait_front(1);
        scale_cb.reserve_back(1);
        const uint32_t scale_hub_l1 = scale_hub_cb.get_read_ptr();
        const uint32_t scale_l1 = scale_cb.get_write_ptr();
        invalidate_l1_cache();
        copy_l1_words(scale_l1, scale_hub_l1, row_tile_bytes);
        for (uint32_t i = 1; i < num_stats; ++i) {
            const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * (first + i));
            const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * (first + i) + 1u);
            noc_async_write(scale_hub_l1, get_noc_addr(dst_x, dst_y, scale_l1), row_tile_bytes);
        }
        if constexpr (num_stats > 1) {
            noc_async_write_barrier();
            for (uint32_t i = 1; i < num_stats; ++i) {
                const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * (first + i));
                const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * (first + i) + 1u);
                noc_semaphore_inc(get_noc_addr(dst_x, dst_y, ready_sem_addr), 1);
            }
            noc_async_atomic_barrier();
        }
        scale_hub_cb.pop_front(1);
        scale_cb.push_back(1);
    } else {
        const uint32_t gather_l1 = gather_cb.get_write_ptr();
        local_stat_cb.wait_front(1);
        noc_async_write(
            local_stat_cb.get_read_ptr(),
            get_noc_addr(hub_x, hub_y, gather_l1 + slot * row_tile_bytes),
            row_tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, arrive_sem_addr), 1);
        noc_async_atomic_barrier();
        local_stat_cb.pop_front(1);

        scale_cb.reserve_back(1);
        noc_semaphore_wait(ready_sem, 1);
        noc_semaphore_set(ready_sem, 0);
        scale_cb.push_back(1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t do_pool = get_compile_time_arg_val(0);
    constexpr uint32_t has_window = get_compile_time_arg_val(1);
    constexpr uint32_t has_prev_slot = get_compile_time_arg_val(2);
    constexpr uint32_t has_rope = get_compile_time_arg_val(3);
    constexpr uint32_t gate_sharded = get_compile_time_arg_val(4);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t row_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t win_slot_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t prev_slot_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t pos_slot_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(12);

    constexpr uint32_t cb_gate_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_g0_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_local_stat_id = get_compile_time_arg_val(15);
    constexpr uint32_t cb_gather_stats_id = get_compile_time_arg_val(16);
    constexpr uint32_t cb_scale_hub_id = get_compile_time_arg_val(17);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(18);
    constexpr uint32_t cb_out_latent_id = get_compile_time_arg_val(19);
    constexpr uint32_t cb_out_rope_id = get_compile_time_arg_val(20);
    constexpr uint32_t cb_win_gate_tile_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_gate_scratch_id = get_compile_time_arg_val(22);
    constexpr uint32_t cb_pos_id = get_compile_time_arg_val(23);
    constexpr uint32_t cb_out_tile_scratch_id = get_compile_time_arg_val(24);
    constexpr uint32_t has_index_key = get_compile_time_arg_val(25);
    constexpr uint32_t has_comp_kv = get_compile_time_arg_val(26);
    constexpr uint32_t has_dense = get_compile_time_arg_val(27);
    constexpr uint32_t has_page_row = get_compile_time_arg_val(28);
    constexpr uint32_t has_key_row = get_compile_time_arg_val(29);
    constexpr uint32_t key_tiles = get_compile_time_arg_val(30);
    constexpr uint32_t block_rows = get_compile_time_arg_val(31);
    constexpr uint32_t idx_slot_bytes = get_compile_time_arg_val(32);
    constexpr uint32_t entry_page_bytes = get_compile_time_arg_val(33);
    constexpr uint32_t page_row_page_bytes = get_compile_time_arg_val(34);
    constexpr uint32_t dense_entry_page_bytes = get_compile_time_arg_val(35);
    constexpr uint32_t page_table_chunk_bytes = get_compile_time_arg_val(36);
    constexpr uint32_t key_latent_sem_id = get_compile_time_arg_val(37);
    constexpr uint32_t key_arrive_sem_id = get_compile_time_arg_val(38);
    constexpr uint32_t key_ready_sem_id = get_compile_time_arg_val(39);
    constexpr uint32_t cb_key_in_id = get_compile_time_arg_val(40);
    constexpr uint32_t cb_key_out_id = get_compile_time_arg_val(41);
    constexpr uint32_t cb_key_tile_id = get_compile_time_arg_val(42);
    constexpr uint32_t cb_dense_tile_id = get_compile_time_arg_val(43);
    constexpr uint32_t cb_idx_id = get_compile_time_arg_val(44);

    constexpr auto gate_args = TensorAccessorArgs<45>();
    constexpr auto win_gate_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto win_slot_args = TensorAccessorArgs<win_gate_args.next_compile_time_args_offset()>();
    constexpr auto prev_slot_args = TensorAccessorArgs<win_slot_args.next_compile_time_args_offset()>();
    constexpr auto out_latent_args = TensorAccessorArgs<prev_slot_args.next_compile_time_args_offset()>();
    constexpr auto out_rope_args = TensorAccessorArgs<out_latent_args.next_compile_time_args_offset()>();
    constexpr auto idx_key_cache_args = TensorAccessorArgs<out_rope_args.next_compile_time_args_offset()>();
    constexpr auto comp_kv_args = TensorAccessorArgs<idx_key_cache_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<comp_kv_args.next_compile_time_args_offset()>();
    constexpr auto entry_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto page_row_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
    constexpr auto dense_kv_args = TensorAccessorArgs<page_row_args.next_compile_time_args_offset()>();
    constexpr auto dense_entry_args = TensorAccessorArgs<dense_kv_args.next_compile_time_args_offset()>();
    constexpr auto key_row_args = TensorAccessorArgs<dense_entry_args.next_compile_time_args_offset()>();

    const uint32_t t = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t gate_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t gate_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t gate_offset_bytes = get_arg_val<uint32_t>(4);
    const uint32_t win_gate_addr = get_arg_val<uint32_t>(5);
    const uint32_t win_slot_addr = get_arg_val<uint32_t>(6);
    const uint32_t prev_slot_addr = get_arg_val<uint32_t>(7);
    const uint32_t out_latent_addr = get_arg_val<uint32_t>(8);
    const uint32_t out_rope_addr = get_arg_val<uint32_t>(9);
    const bool do_rope = get_arg_val<uint32_t>(10) != 0;
    const uint32_t hub_noc_x = get_arg_val<uint32_t>(11);
    const uint32_t hub_noc_y = get_arg_val<uint32_t>(12);
    const uint32_t idx_key_cache_addr = get_arg_val<uint32_t>(13);
    const uint32_t comp_kv_addr = get_arg_val<uint32_t>(14);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(15);
    const uint32_t entry_addr = get_arg_val<uint32_t>(16);
    const uint32_t page_row_addr = get_arg_val<uint32_t>(17);
    const uint32_t dense_kv_addr = get_arg_val<uint32_t>(18);
    const uint32_t dense_entry_addr = get_arg_val<uint32_t>(19);
    const uint32_t key_row_addr = get_arg_val<uint32_t>(20);
    const bool is_key_core = get_arg_val<uint32_t>(21) != 0;
    const uint32_t key_idx = get_arg_val<uint32_t>(22);

    constexpr uint32_t tile_rows = 32;
    constexpr bool reads_entry = has_index_key != 0 || has_comp_kv != 0;
    constexpr uint32_t first_key_core = head_tiles - key_tiles;

    auto get_gate_noc_addr = [&]() -> uint64_t {
        if constexpr (gate_sharded != 0) {
            return get_noc_addr(gate_noc_x, gate_noc_y, gate_addr + gate_offset_bytes);
        } else {
            const auto gate_acc = TensorAccessor(gate_args, gate_addr);
            return gate_acc.get_noc_addr(0, gate_offset_bytes);
        }
    };

    if constexpr (do_pool == 0) {
        CircularBuffer win_gate_tile_cb(cb_win_gate_tile_id);
        CircularBuffer gate_scratch_cb(cb_gate_scratch_id);
        CircularBuffer pos_cb(cb_pos_id);

        pos_cb.reserve_back(2);
        gate_scratch_cb.reserve_back(1);
        win_gate_tile_cb.reserve_back(1);

        const uint32_t pos_l1 = pos_cb.get_write_ptr();
        const uint32_t gate_scratch_l1 = gate_scratch_cb.get_write_ptr();
        const uint32_t tile_l1 = win_gate_tile_cb.get_write_ptr();

        const auto win_slot_acc = TensorAccessor(win_slot_args, win_slot_addr);
        noc_async_read(win_slot_acc.get_noc_addr(0), pos_l1, win_slot_page_bytes);
        noc_async_read(get_gate_noc_addr(), gate_scratch_l1, row_tile_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();

        const uint32_t win_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[0];
        const auto win_gate_acc = TensorAccessor(win_gate_args, win_gate_addr);
        const uint64_t tile_noc = win_gate_acc.get_noc_addr((win_slot_idx / tile_rows) * head_tiles + t);
        noc_async_read(tile_noc, tile_l1, cache_tile_bytes);
        noc_async_read_barrier();

        patch_tile_row(tile_l1, gate_scratch_l1, win_slot_idx % tile_rows);
        noc_async_write(tile_l1, tile_noc, cache_tile_bytes);
        noc_async_write_barrier();
        return;
    }

    CircularBuffer out_tile_cb(cb_out_tile_scratch_id);
    out_tile_cb.reserve_back(1);
    const uint32_t out_tile_l1 = out_tile_cb.get_write_ptr();
    zero_l1_async(out_tile_l1, cache_tile_bytes);

    if constexpr (has_window != 0) {
        CircularBuffer gate_cb(cb_gate_id);
        CircularBuffer g0_cb(cb_g0_id);
        CircularBuffer gate_scratch_cb(cb_gate_scratch_id);
        CircularBuffer win_gate_tile_cb(cb_win_gate_tile_id);
        CircularBuffer pos_cb(cb_pos_id);

        gate_cb.reserve_back(1);
        g0_cb.reserve_back(1);
        gate_scratch_cb.reserve_back(1);
        win_gate_tile_cb.reserve_back(1);
        pos_cb.reserve_back(2);

        const uint32_t gate_l1 = gate_cb.get_write_ptr();
        const uint32_t g0_l1 = g0_cb.get_write_ptr();
        const uint32_t gate_scratch_l1 = gate_scratch_cb.get_write_ptr();
        const uint32_t tile_l1 = win_gate_tile_cb.get_write_ptr();
        const uint32_t pos_l1 = pos_cb.get_write_ptr();

        noc_async_read(get_gate_noc_addr(), gate_l1, row_tile_bytes);
        const auto win_slot_acc = TensorAccessor(win_slot_args, win_slot_addr);
        noc_async_read(win_slot_acc.get_noc_addr(0), pos_l1, win_slot_page_bytes);
        if constexpr (has_prev_slot != 0) {
            const auto prev_slot_acc = TensorAccessor(prev_slot_args, prev_slot_addr);
            noc_async_read(prev_slot_acc.get_noc_addr(0), pos_l1 + pos_slot_bytes, prev_slot_page_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();

        copy_l1_words(gate_scratch_l1, gate_l1, row_tile_bytes);
        gate_cb.push_back(1);

        const uint32_t win_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[0];
        uint32_t prev_slot_idx = 1u - win_slot_idx;
        if constexpr (has_prev_slot != 0) {
            prev_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1 + pos_slot_bytes)[0];
        }

        const auto win_gate_acc = TensorAccessor(win_gate_args, win_gate_addr);
        const uint32_t prev_tile_row = prev_slot_idx / tile_rows;
        const uint32_t win_tile_row = win_slot_idx / tile_rows;
        const uint64_t prev_tile_noc = win_gate_acc.get_noc_addr(prev_tile_row * head_tiles + t);
        noc_async_read(prev_tile_noc, tile_l1, cache_tile_bytes);
        noc_async_read_barrier();

        extract_tile_row(g0_l1, tile_l1, prev_slot_idx % tile_rows);
        g0_cb.push_back(1);

        const uint64_t win_tile_noc =
            (win_tile_row == prev_tile_row) ? prev_tile_noc : win_gate_acc.get_noc_addr(win_tile_row * head_tiles + t);
        if (win_tile_row != prev_tile_row) {
            noc_async_read(win_tile_noc, tile_l1, cache_tile_bytes);
            noc_async_read_barrier();
        }
        patch_tile_row(tile_l1, gate_scratch_l1, win_slot_idx % tile_rows);
        noc_async_write(tile_l1, win_tile_noc, cache_tile_bytes);
        noc_async_write_barrier();
    } else {
        noc_async_read_barrier();
    }

    // Where the entry lands: row in_block_row of physical block phys_block in idx_key_cache / comp_kv, and
    // row dense_row of dense_kv (< 0: not written). The tiles the rows patch into are read while compute pools.
    uint32_t phys_block = 0;
    uint32_t in_block_row = 0;
    int32_t dense_row = -1;
    uint32_t key_tile_l1 = 0;
    uint64_t key_tile_noc = 0;
    uint32_t dense_tile_l1 = 0;
    uint64_t dense_tile_noc = 0;
    if constexpr (reads_entry || has_dense != 0) {
        CircularBuffer idx_cb(cb_idx_id);
        idx_cb.reserve_back(1);
        const uint32_t idx_l1 = idx_cb.get_write_ptr();
        if constexpr (reads_entry) {
            noc_async_read(TensorAccessor(entry_args, entry_addr).get_noc_addr(0), idx_l1, entry_page_bytes);
            if constexpr (has_page_row != 0) {
                noc_async_read(
                    TensorAccessor(page_row_args, page_row_addr).get_noc_addr(0),
                    idx_l1 + idx_slot_bytes,
                    page_row_page_bytes);
            }
        }
        if constexpr (has_dense != 0) {
            noc_async_read(
                TensorAccessor(dense_entry_args, dense_entry_addr).get_noc_addr(0),
                idx_l1 + 2u * idx_slot_bytes,
                dense_entry_page_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();

        if constexpr (reads_entry) {
            const uint32_t entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1)[0];
            uint32_t table_row = 0;
            if constexpr (has_page_row != 0) {
                table_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1 + idx_slot_bytes)[0];
            }
            // Only the aligned piece of the page-table row holding the entry's block.
            const uint32_t block_byte = (entry / block_rows) * sizeof(uint32_t);
            const uint32_t chunk_byte = block_byte & ~(page_table_chunk_bytes - 1u);
            const uint32_t table_l1 = idx_l1 + 3u * idx_slot_bytes;
            noc_async_read(
                TensorAccessor(page_table_args, page_table_addr).get_noc_addr(table_row, chunk_byte),
                table_l1,
                page_table_chunk_bytes);
            noc_async_read_barrier();
            invalidate_l1_cache();
            phys_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1 + block_byte - chunk_byte)[0];
            in_block_row = entry % block_rows;
        }
        if constexpr (has_index_key != 0) {
            if (is_key_core) {
                CircularBuffer key_tile_cb(cb_key_tile_id);
                key_tile_cb.reserve_back(1);
                key_tile_l1 = key_tile_cb.get_write_ptr();
                constexpr uint32_t block_tiles = (block_rows / tile_rows) * key_tiles;
                key_tile_noc =
                    TensorAccessor(idx_key_cache_args, idx_key_cache_addr)
                        .get_noc_addr(phys_block * block_tiles + (in_block_row / tile_rows) * key_tiles + key_idx);
                noc_async_read(key_tile_noc, key_tile_l1, cache_tile_bytes);
            }
        }
        if constexpr (has_dense != 0) {
            dense_row = reinterpret_cast<volatile tt_l1_ptr int32_t*>(idx_l1 + 2u * idx_slot_bytes)[0];
            if (dense_row >= 0) {
                CircularBuffer dense_tile_cb(cb_dense_tile_id);
                dense_tile_cb.reserve_back(1);
                dense_tile_l1 = dense_tile_cb.get_write_ptr();
                dense_tile_noc = TensorAccessor(dense_kv_args, dense_kv_addr)
                                     .get_noc_addr((static_cast<uint32_t>(dense_row) / tile_rows) * head_tiles + t);
                noc_async_read(dense_tile_noc, dense_tile_l1, cache_tile_bytes);
            }
        }
    }

    // Cross-core RMSNorm gather on hub core 0 and distribution of scale.
    gather_rms_scale<head_tiles, row_tile_bytes>(
        t,
        0,
        hub_noc_x,
        hub_noc_y,
        cb_local_stat_id,
        cb_gather_stats_id,
        cb_scale_hub_id,
        cb_scale_id,
        arrive_sem_id,
        ready_sem_id);

    // Write latent (TILE 32x32) and optional roped_row (ROW_MAJOR [1, 1, 1, Dh]).
    CircularBuffer out_latent_cb(cb_out_latent_id);
    const auto out_latent_acc = TensorAccessor(out_latent_args, out_latent_addr);
    out_latent_cb.wait_front(1);
    const uint32_t latent_l1 = out_latent_cb.get_read_ptr();

    // Latent tile t is K-tile t of every key core's matmul input.
    if constexpr (has_index_key != 0) {
        const uint32_t key_in_l1 = CircularBuffer(cb_key_in_id).get_write_ptr() + t * row_tile_bytes;
        const uint32_t key_latent_sem_addr = get_semaphore(key_latent_sem_id);
        for (uint32_t n = 0; n < key_tiles; ++n) {
            const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * (first_key_core + n));
            const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * (first_key_core + n) + 1u);
            noc_async_write(latent_l1, get_noc_addr(dst_x, dst_y, key_in_l1), row_tile_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t n = 0; n < key_tiles; ++n) {
            const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * (first_key_core + n));
            const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * (first_key_core + n) + 1u);
            noc_semaphore_inc(get_noc_addr(dst_x, dst_y, key_latent_sem_addr), 1);
        }
    }

    patch_tile_row(out_tile_l1, latent_l1, 0);
    noc_async_write(out_tile_l1, out_latent_acc.get_noc_addr(t), cache_tile_bytes);

    if constexpr (has_rope != 0) {
        const auto out_rope_acc = TensorAccessor(out_rope_args, out_rope_addr);
        CircularBuffer out_rope_cb(cb_out_rope_id);
        uint32_t row_l1 = latent_l1;
        if (do_rope) {
            out_rope_cb.wait_front(1);
            row_l1 = out_rope_cb.get_read_ptr();
        }
        noc_async_write(row_l1, out_rope_acc.get_noc_addr(0, t * row_tile_bytes), row_tile_bytes);
        if constexpr (has_comp_kv != 0) {
            noc_async_write(
                row_l1,
                TensorAccessor(comp_kv_args, comp_kv_addr)
                    .get_noc_addr(phys_block * block_rows + in_block_row, t * row_tile_bytes),
                row_tile_bytes);
        }
        if constexpr (has_dense != 0) {
            if (dense_row >= 0) {
                noc_async_read_barrier();
                invalidate_l1_cache();
                patch_tile_row(dense_tile_l1, row_l1, static_cast<uint32_t>(dense_row) % tile_rows);
                noc_async_write(dense_tile_l1, dense_tile_noc, cache_tile_bytes);
            }
        }
        noc_async_write_barrier();
        if (do_rope) {
            out_rope_cb.pop_front(1);
        }
        out_latent_cb.pop_front(1);
    } else {
        noc_async_write_barrier();
        out_latent_cb.pop_front(1);
    }

    // Key core key_idx: the K-reduction waits for every latent tile; the key's RMS stats gather on key core 0.
    if constexpr (has_index_key != 0) {
        if (is_key_core) {
            CircularBuffer key_in_cb(cb_key_in_id);
            volatile tt_l1_ptr uint32_t* key_latent_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(key_latent_sem_id));
            key_in_cb.reserve_back(head_tiles);
            noc_semaphore_wait(key_latent_sem, head_tiles);
            noc_semaphore_set(key_latent_sem, 0);
            key_in_cb.push_back(head_tiles);

            gather_rms_scale<key_tiles, row_tile_bytes>(
                key_idx,
                first_key_core,
                get_common_arg_val<uint32_t>(2u * first_key_core),
                get_common_arg_val<uint32_t>(2u * first_key_core + 1u),
                cb_local_stat_id,
                cb_gather_stats_id,
                cb_scale_hub_id,
                cb_scale_id,
                key_arrive_sem_id,
                key_ready_sem_id);

            CircularBuffer key_out_cb(cb_key_out_id);
            key_out_cb.wait_front(1);
            const uint32_t key_l1 = key_out_cb.get_read_ptr();
            noc_async_read_barrier();
            invalidate_l1_cache();
            patch_tile_row(key_tile_l1, key_l1, in_block_row % tile_rows);
            noc_async_write(key_tile_l1, key_tile_noc, cache_tile_bytes);
            if constexpr (has_key_row != 0) {
                noc_async_write(
                    key_l1,
                    TensorAccessor(key_row_args, key_row_addr).get_noc_addr(0, key_idx * row_tile_bytes),
                    row_tile_bytes);
            }
            noc_async_write_barrier();
            key_out_cb.pop_front(1);
        }
        noc_async_atomic_barrier();
    }
}
