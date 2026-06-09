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
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(0);

    const uint32_t trans_mat_tile_size = get_tile_size(trans_mat_cb_id);
    if (trans_mat_tile_size == 4096) {
        fill_rotate_half_trans_mat_fp32(trans_mat_cb_id);
    } else if (trans_mat_tile_size == 2048) {
        fill_rotate_half_trans_mat_bf16(trans_mat_cb_id);
    } else {
        fill_rotate_half_trans_mat_bfp8(trans_mat_cb_id);
    }
}
