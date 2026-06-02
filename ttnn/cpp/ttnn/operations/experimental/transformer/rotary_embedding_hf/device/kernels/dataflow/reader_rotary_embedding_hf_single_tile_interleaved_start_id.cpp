// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline void fill_rotate_half_trans_mat_bfp8(uint32_t cb_id) {
    constexpr uint32_t onetile = 1;
    cb_reserve_back(cb_id, onetile);
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1088; ++i) {
        p[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        p[16 + r] = 127;
        p[320 + r * 16 + r] = 0x40;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        p[32 + r] = 127;
        p[576 + r * 16 + r] = 0xC0;
    }
    cb_push_back(cb_id, onetile);
}

inline void fill_rotate_half_trans_mat_bf16(uint32_t cb_id) {
    constexpr uint32_t onetile = 1;
    constexpr uint16_t one_bf16 = 0x3F80;
    constexpr uint16_t neg_one_bf16 = 0xBF80;
    constexpr uint32_t face_elems = 16 * 16;

    cb_reserve_back(cb_id, onetile);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_bf16;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_bf16;
    }
    cb_push_back(cb_id, onetile);
}

inline void fill_rotate_half_trans_mat_fp32(uint32_t cb_id) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t one_fp32 = 0x3F800000;
    constexpr uint32_t neg_one_fp32 = 0xBF800000;
    constexpr uint32_t face_elems = 16 * 16;

    cb_reserve_back(cb_id, onetile);
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_fp32;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_fp32;
    }
    cb_push_back(cb_id, onetile);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t cos_addr = get_arg_val<uint32_t>(1);
    uint32_t sin_addr = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t start_row_id = get_arg_val<uint32_t>(5);
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t HtWt = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr, input_tile_bytes);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    const uint32_t trans_mat_tile_size = get_tile_size(trans_mat_cb_id);
    if (trans_mat_tile_size == 4096) {
        fill_rotate_half_trans_mat_fp32(trans_mat_cb_id);
    } else if (trans_mat_tile_size == 2048) {
        fill_rotate_half_trans_mat_bf16(trans_mat_cb_id);
    } else {
        fill_rotate_half_trans_mat_bfp8(trans_mat_cb_id);
    }

    uint32_t input_curr_id = start_id;
    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;

    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_reserve_back(input_cb_id, onetile);
        uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
        noc_async_read_page(input_curr_id, s0, input_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(input_cb_id, onetile);
        input_curr_id++;

        cb_reserve_back(sin_cb_id, onetile);
        {
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
            noc_async_read_page(cos_sin_curr_id, s2, sin_l1_write_addr);
            noc_async_read_barrier();
        }
        cb_push_back(sin_cb_id, onetile);

        cb_reserve_back(cos_cb_id, onetile);
        {
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            noc_async_read_page(cos_sin_curr_id, s1, cos_l1_write_addr);
            noc_async_read_barrier();
        }
        cb_push_back(cos_cb_id, onetile);
        cos_sin_curr_id++;

        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
