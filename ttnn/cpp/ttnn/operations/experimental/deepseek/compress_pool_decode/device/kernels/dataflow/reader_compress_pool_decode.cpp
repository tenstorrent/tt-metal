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
    constexpr uint32_t kv_sharded = get_compile_time_arg_val(4);
    constexpr uint32_t norm_weight_is_tile = get_compile_time_arg_val(5);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t row_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t win_slot_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t prev_slot_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t pos_slot_bytes = get_compile_time_arg_val(11);

    constexpr uint32_t cb_kv_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_k0_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_gamma_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_cs_id = get_compile_time_arg_val(15);
    constexpr uint32_t cb_win_kv_tile_id = get_compile_time_arg_val(16);
    constexpr uint32_t cb_kv_scratch_id = get_compile_time_arg_val(17);
    constexpr uint32_t cb_pos_id = get_compile_time_arg_val(18);
    constexpr uint32_t cb_raw_scratch_id = get_compile_time_arg_val(19);

    constexpr auto kv_args = TensorAccessorArgs<20>();
    constexpr auto win_kv_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto win_slot_args = TensorAccessorArgs<win_kv_args.next_compile_time_args_offset()>();
    constexpr auto prev_slot_args = TensorAccessorArgs<win_slot_args.next_compile_time_args_offset()>();
    constexpr auto norm_weight_args = TensorAccessorArgs<prev_slot_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<norm_weight_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const uint32_t t = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t kv_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t kv_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t kv_offset_bytes = get_arg_val<uint32_t>(4);
    const uint32_t win_kv_addr = get_arg_val<uint32_t>(5);
    const uint32_t win_slot_addr = get_arg_val<uint32_t>(6);
    const uint32_t prev_slot_addr = get_arg_val<uint32_t>(7);
    const uint32_t norm_weight_addr = get_arg_val<uint32_t>(8);
    const uint32_t cos_addr = get_arg_val<uint32_t>(9);
    const uint32_t sin_addr = get_arg_val<uint32_t>(10);
    const bool do_rope = get_arg_val<uint32_t>(11) != 0;
    const uint32_t rope_tile_idx = get_arg_val<uint32_t>(12);

    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;

    auto get_kv_noc_addr = [&]() -> uint64_t {
        if constexpr (kv_sharded != 0) {
            return get_noc_addr(kv_noc_x, kv_noc_y, kv_addr + kv_offset_bytes);
        } else {
            const auto kv_acc = TensorAccessor(kv_args, kv_addr);
            return kv_acc.get_noc_addr(0, kv_offset_bytes);
        }
    };

    if constexpr (do_pool == 0) {
        CircularBuffer win_kv_tile_cb(cb_win_kv_tile_id);
        CircularBuffer kv_scratch_cb(cb_kv_scratch_id);
        CircularBuffer pos_cb(cb_pos_id);

        pos_cb.reserve_back(2);
        kv_scratch_cb.reserve_back(1);
        win_kv_tile_cb.reserve_back(1);

        const uint32_t pos_l1 = pos_cb.get_write_ptr();
        const uint32_t kv_scratch_l1 = kv_scratch_cb.get_write_ptr();
        const uint32_t tile_l1 = win_kv_tile_cb.get_write_ptr();

        const auto win_slot_acc = TensorAccessor(win_slot_args, win_slot_addr);
        noc_async_read(win_slot_acc.get_noc_addr(0), pos_l1, win_slot_page_bytes);
        noc_async_read(get_kv_noc_addr(), kv_scratch_l1, row_tile_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();

        const uint32_t win_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[0];
        const auto win_kv_acc = TensorAccessor(win_kv_args, win_kv_addr);
        const uint64_t tile_noc = win_kv_acc.get_noc_addr((win_slot_idx / tile_rows) * head_tiles + t);
        noc_async_read(tile_noc, tile_l1, cache_tile_bytes);
        noc_async_read_barrier();

        patch_tile_row(tile_l1, kv_scratch_l1, win_slot_idx % tile_rows);
        noc_async_write(tile_l1, tile_noc, cache_tile_bytes);
        noc_async_write_barrier();
        return;
    }

    CircularBuffer kv_cb(cb_kv_id);
    CircularBuffer gamma_cb(cb_gamma_id);
    CircularBuffer raw_cb(cb_raw_scratch_id);

    kv_cb.reserve_back(1);
    gamma_cb.reserve_back(1);
    raw_cb.reserve_back(6);

    const uint32_t kv_l1 = kv_cb.get_write_ptr();
    const uint32_t gamma_l1 = gamma_cb.get_write_ptr();
    const uint32_t raw_l1 = raw_cb.get_write_ptr();

    noc_async_read(get_kv_noc_addr(), kv_l1, row_tile_bytes);

    const auto norm_acc = TensorAccessor(norm_weight_args, norm_weight_addr);
    if constexpr (norm_weight_is_tile != 0) {
        noc_async_read(norm_acc.get_noc_addr(t, 0), raw_l1, 64u);
        noc_async_read(norm_acc.get_noc_addr(t, face_bytes), raw_l1 + 64u, 64u);
    } else {
        noc_async_read(norm_acc.get_noc_addr(0, t * row_tile_bytes), gamma_l1, row_tile_bytes);
    }

    uint32_t cs_l1 = 0;
    if constexpr (has_rope != 0) {
        if (do_rope) {
            CircularBuffer cs_cb(cb_cs_id);
            cs_cb.reserve_back(2);
            cs_l1 = cs_cb.get_write_ptr();
            const auto cos_acc = TensorAccessor(cos_args, cos_addr);
            const auto sin_acc = TensorAccessor(sin_args, sin_addr);
            noc_async_read(cos_acc.get_noc_addr(rope_tile_idx, 0), raw_l1 + 128u, 64u);
            noc_async_read(cos_acc.get_noc_addr(rope_tile_idx, face_bytes), raw_l1 + 192u, 64u);
            noc_async_read(sin_acc.get_noc_addr(rope_tile_idx, 0), raw_l1 + 256u, 64u);
            noc_async_read(sin_acc.get_noc_addr(rope_tile_idx, face_bytes), raw_l1 + 320u, 64u);
        }
    }

    uint32_t pos_l1 = 0;
    uint32_t kv_scratch_l1 = 0;
    if constexpr (has_window != 0) {
        CircularBuffer pos_cb(cb_pos_id);
        CircularBuffer kv_scratch_cb(cb_kv_scratch_id);
        pos_cb.reserve_back(2);
        kv_scratch_cb.reserve_back(1);
        pos_l1 = pos_cb.get_write_ptr();
        kv_scratch_l1 = kv_scratch_cb.get_write_ptr();

        const auto win_slot_acc = TensorAccessor(win_slot_args, win_slot_addr);
        noc_async_read(win_slot_acc.get_noc_addr(0), pos_l1, win_slot_page_bytes);
        if constexpr (has_prev_slot != 0) {
            const auto prev_slot_acc = TensorAccessor(prev_slot_args, prev_slot_addr);
            noc_async_read(prev_slot_acc.get_noc_addr(0), pos_l1 + pos_slot_bytes, prev_slot_page_bytes);
        }
    }

    noc_async_read_barrier();

    if constexpr (has_window != 0) {
        copy_l1_words(kv_scratch_l1, kv_l1, row_tile_bytes);
    }
    kv_cb.push_back(1);

    if constexpr (norm_weight_is_tile != 0) {
        copy_l1_words(gamma_l1, raw_l1, face_row_bytes);
        copy_l1_words(gamma_l1 + face_row_bytes, raw_l1 + 64u, face_row_bytes);
    }
    gamma_cb.push_back(1);

    if constexpr (has_rope != 0) {
        if (do_rope) {
            CircularBuffer cs_cb(cb_cs_id);
            copy_l1_words(cs_l1, raw_l1 + 128u, face_row_bytes);
            copy_l1_words(cs_l1 + face_row_bytes, raw_l1 + 192u, face_row_bytes);
            copy_l1_words(cs_l1 + 2u * face_row_bytes, raw_l1 + 256u, face_row_bytes);
            copy_l1_words(cs_l1 + 3u * face_row_bytes, raw_l1 + 320u, face_row_bytes);
            cs_cb.push_back(2);
        }
    }

    if constexpr (has_window != 0) {
        invalidate_l1_cache();
        const uint32_t win_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[0];
        uint32_t prev_slot_idx = 1u - win_slot_idx;
        if constexpr (has_prev_slot != 0) {
            prev_slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1 + pos_slot_bytes)[0];
        }

        CircularBuffer k0_cb(cb_k0_id);
        CircularBuffer win_kv_tile_cb(cb_win_kv_tile_id);
        k0_cb.reserve_back(1);
        win_kv_tile_cb.reserve_back(1);
        const uint32_t k0_l1 = k0_cb.get_write_ptr();
        const uint32_t tile_l1 = win_kv_tile_cb.get_write_ptr();

        const auto win_kv_acc = TensorAccessor(win_kv_args, win_kv_addr);
        const uint32_t prev_tile_row = prev_slot_idx / tile_rows;
        const uint32_t win_tile_row = win_slot_idx / tile_rows;
        const uint64_t prev_tile_noc = win_kv_acc.get_noc_addr(prev_tile_row * head_tiles + t);
        noc_async_read(prev_tile_noc, tile_l1, cache_tile_bytes);
        noc_async_read_barrier();

        extract_tile_row(k0_l1, tile_l1, prev_slot_idx % tile_rows);
        k0_cb.push_back(1);

        const uint64_t win_tile_noc =
            (win_tile_row == prev_tile_row) ? prev_tile_noc : win_kv_acc.get_noc_addr(win_tile_row * head_tiles + t);
        if (win_tile_row != prev_tile_row) {
            noc_async_read(win_tile_noc, tile_l1, cache_tile_bytes);
            noc_async_read_barrier();
        }
        patch_tile_row(tile_l1, kv_scratch_l1, win_slot_idx % tile_rows);
        noc_async_write(tile_l1, win_tile_noc, cache_tile_bytes);
        noc_async_write_barrier();
    }
}
