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

    constexpr auto gate_args = TensorAccessorArgs<25>();
    constexpr auto win_gate_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto win_slot_args = TensorAccessorArgs<win_gate_args.next_compile_time_args_offset()>();
    constexpr auto prev_slot_args = TensorAccessorArgs<win_slot_args.next_compile_time_args_offset()>();
    constexpr auto out_latent_args = TensorAccessorArgs<prev_slot_args.next_compile_time_args_offset()>();
    constexpr auto out_rope_args = TensorAccessorArgs<out_latent_args.next_compile_time_args_offset()>();

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

    constexpr uint32_t tile_rows = 32;

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

    // Cross-core RMSNorm gather on hub core 0 and distribution of scale.
    CircularBuffer local_stat_cb(cb_local_stat_id);
    CircularBuffer gather_cb(cb_gather_stats_id);
    CircularBuffer scale_cb(cb_scale_id);

    const uint32_t arrive_sem_addr = get_semaphore(arrive_sem_id);
    volatile tt_l1_ptr uint32_t* arrive_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem_addr);
    const uint32_t ready_sem_addr = get_semaphore(ready_sem_id);
    volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);

    if (t == 0) {
        gather_cb.reserve_back(head_tiles);
        const uint32_t gather_l1 = gather_cb.get_write_ptr();
        local_stat_cb.wait_front(1);
        invalidate_l1_cache();
        copy_l1_words(gather_l1, local_stat_cb.get_read_ptr(), row_tile_bytes);
        local_stat_cb.pop_front(1);
        if constexpr (head_tiles > 1) {
            noc_semaphore_wait(arrive_sem, head_tiles - 1);
            noc_semaphore_set(arrive_sem, 0);
        }
        gather_cb.push_back(head_tiles);

        CircularBuffer scale_hub_cb(cb_scale_hub_id);
        scale_hub_cb.wait_front(1);
        scale_cb.reserve_back(1);
        const uint32_t scale_hub_l1 = scale_hub_cb.get_read_ptr();
        const uint32_t scale_l1 = scale_cb.get_write_ptr();
        invalidate_l1_cache();
        copy_l1_words(scale_l1, scale_hub_l1, row_tile_bytes);
        for (uint32_t i = 1; i < head_tiles; ++i) {
            const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * i);
            const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * i + 1u);
            noc_async_write(scale_hub_l1, get_noc_addr(dst_x, dst_y, scale_l1), row_tile_bytes);
        }
        if constexpr (head_tiles > 1) {
            noc_async_write_barrier();
            for (uint32_t i = 1; i < head_tiles; ++i) {
                const uint32_t dst_x = get_common_arg_val<uint32_t>(2u * i);
                const uint32_t dst_y = get_common_arg_val<uint32_t>(2u * i + 1u);
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
            get_noc_addr(hub_noc_x, hub_noc_y, gather_l1 + t * row_tile_bytes),
            row_tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_noc_x, hub_noc_y, arrive_sem_addr), 1);
        noc_async_atomic_barrier();
        local_stat_cb.pop_front(1);

        scale_cb.reserve_back(1);
        noc_semaphore_wait(ready_sem, 1);
        noc_semaphore_set(ready_sem, 0);
        scale_cb.push_back(1);
    }

    // Write latent (TILE 32x32) and optional roped_row (ROW_MAJOR [1, 1, 1, Dh]).
    CircularBuffer out_latent_cb(cb_out_latent_id);
    const auto out_latent_acc = TensorAccessor(out_latent_args, out_latent_addr);
    out_latent_cb.wait_front(1);
    const uint32_t latent_l1 = out_latent_cb.get_read_ptr();
    patch_tile_row(out_tile_l1, latent_l1, 0);
    noc_async_write(out_tile_l1, out_latent_acc.get_noc_addr(t), cache_tile_bytes);

    if constexpr (has_rope != 0) {
        const auto out_rope_acc = TensorAccessor(out_rope_args, out_rope_addr);
        if (do_rope) {
            CircularBuffer out_rope_cb(cb_out_rope_id);
            out_rope_cb.wait_front(1);
            noc_async_write(
                out_rope_cb.get_read_ptr(), out_rope_acc.get_noc_addr(0, t * row_tile_bytes), row_tile_bytes);
            noc_async_write_barrier();
            out_rope_cb.pop_front(1);
            out_latent_cb.pop_front(1);
        } else {
            noc_async_write(latent_l1, out_rope_acc.get_noc_addr(0, t * row_tile_bytes), row_tile_bytes);
            noc_async_write_barrier();
            out_latent_cb.pop_front(1);
        }
    } else {
        noc_async_write_barrier();
        out_latent_cb.pop_front(1);
    }
}
