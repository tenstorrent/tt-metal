// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

inline void fill_rotate_half_trans_mat_bfp8(CircularBuffer& cb) {
    constexpr uint32_t onetile = 1;
    cb.reserve_back(onetile);
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb.get_write_ptr());
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
    cb.push_back(onetile);
}

inline void fill_rotate_half_trans_mat_bf16(CircularBuffer& cb) {
    constexpr uint32_t onetile = 1;
    constexpr uint16_t one_bf16 = 0x3F80;
    constexpr uint16_t neg_one_bf16 = 0xBF80;
    constexpr uint32_t face_elems = 16 * 16;

    cb.reserve_back(onetile);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_bf16;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_bf16;
    }
    cb.push_back(onetile);
}

inline void fill_rotate_half_trans_mat_fp32(CircularBuffer& cb) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t one_fp32 = 0x3F800000;
    constexpr uint32_t neg_one_fp32 = 0xBF800000;
    constexpr uint32_t face_elems = 16 * 16;

    cb.reserve_back(onetile);
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_fp32;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_fp32;
    }
    cb.push_back(onetile);
}

void kernel_main() {
    Noc noc;

    uint32_t cos_addr = get_arg_val<uint32_t>(0);
    uint32_t sin_addr = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t start_row_id = get_arg_val<uint32_t>(3);
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t HtWt = get_compile_time_arg_val(5);
    constexpr auto cos_args = TensorAccessorArgs<6>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_trans_mat(trans_mat_cb_id);

    cb_input.reserve_back(num_rows);
    cb_input.push_back(num_rows);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    const uint32_t trans_mat_tile_size = get_tile_size(trans_mat_cb_id);
    if (trans_mat_tile_size == 4096) {
        fill_rotate_half_trans_mat_fp32(cb_trans_mat);
    } else if (trans_mat_tile_size == 2048) {
        fill_rotate_half_trans_mat_bf16(cb_trans_mat);
    } else {
        fill_rotate_half_trans_mat_bfp8(cb_trans_mat);
    }

    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;
    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_sin.reserve_back(onetile);
        {
            uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
        }
        cb_sin.push_back(onetile);

        cb_cos.reserve_back(onetile);
        {
            uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
        }
        cb_cos.push_back(onetile);
        cos_sin_curr_id++;

        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
