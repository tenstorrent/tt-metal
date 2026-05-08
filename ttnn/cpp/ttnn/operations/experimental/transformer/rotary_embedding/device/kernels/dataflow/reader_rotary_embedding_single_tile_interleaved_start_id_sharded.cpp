// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sharded reader for the head_dim == TILE_WIDTH (Wt == 1) path of
// ttnn.experimental.rotary_embedding. Input is already in L1 via the input CB's
// globally-allocated address; this kernel only fetches cos/sin tiles from DRAM
// and emits the rotate_half transformation matrix into an L1 CB once.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Same tile encodes [[0, I_16], [-I_16, 0]] in BFP8_B layout. Tile bytes:
//   0..63   : 4 face exponent sections, 16 bytes each (face0, face1, face2, face3)
//   64..1087: 4 face mantissa sections, 256 bytes each (16 rows x 16 cols per face)
// Each mantissa byte: bit 7 = sign, bits 6..0 = 7-bit magnitude. Decoding uses
// the shared row exponent; with shared_exp = 127, a mantissa byte of 0x40
// decodes to +1.0 and 0xC0 decodes to -1.0.
inline void fill_rotate_half_trans_mat_bfp8(uint32_t cb_id) {
    constexpr uint32_t onetile = 1;
    cb_reserve_back(cb_id, onetile);
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    // Zero the whole tile (1088 bytes).
    for (uint32_t i = 0; i < 1088; ++i) {
        p[i] = 0;
    }
    // Face 1 (rows 0-15, cols 16-31) = +I: exp=127 per row, diagonal mantissa=0x40.
    for (uint32_t r = 0; r < 16; ++r) {
        p[16 + r] = 127;
        p[320 + r * 16 + r] = 0x40;
    }
    // Face 2 (rows 16-31, cols 0-15) = -I: exp=127 per row, diagonal mantissa=0xC0.
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

void kernel_main() {
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

    // Sharded input: present in L1 already; just publish to the consumer.
    cb_reserve_back(input_cb_id, num_rows);
    cb_push_back(input_cb_id, num_rows);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    if (get_tile_size(trans_mat_cb_id) == 2048) {
        fill_rotate_half_trans_mat_bf16(trans_mat_cb_id);
    } else {
        fill_rotate_half_trans_mat_bfp8(trans_mat_cb_id);
    }

    uint32_t cos_sin_curr_id = cos_sin_start_id;

#ifdef DECODE_MODE
    cb_reserve_back(sin_cb_id, onetile);
    cb_reserve_back(cos_cb_id, onetile);
    {
        uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
        uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
        noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
        noc_async_read_tile(cos_sin_curr_id, s1, cos_l1_write_addr);
        noc_async_read_barrier();
    }
    cb_push_back(sin_cb_id, onetile);
    cb_push_back(cos_cb_id, onetile);
#else
    uint32_t ht = start_row_id;
    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_reserve_back(sin_cb_id, onetile);
        {
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
            noc_async_read_barrier();
        }
        cb_push_back(sin_cb_id, onetile);

        cb_reserve_back(cos_cb_id, onetile);
        {
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s1, cos_l1_write_addr);
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
#endif
}
