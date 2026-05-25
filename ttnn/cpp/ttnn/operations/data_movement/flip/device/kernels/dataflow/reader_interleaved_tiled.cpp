// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rank = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr uint32_t TILE_H = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t TILE_W = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t FACE_H = get_named_compile_time_arg_val("face_height");
    constexpr uint32_t FACE_W = get_named_compile_time_arg_val("face_width");
    constexpr uint32_t logical_H = get_named_compile_time_arg_val("logical_H");
    constexpr uint32_t logical_W = get_named_compile_time_arg_val("logical_W");
    constexpr bool is_bfp8 = get_named_compile_time_arg_val("is_bfp8") == 1;
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t NUM_FACES_H = TILE_H / FACE_H;
    constexpr uint32_t NUM_FACES_W = TILE_W / FACE_W;
    constexpr uint32_t FACE_HW_BYTES = FACE_H * FACE_W * element_size;
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_W * element_size;
    constexpr uint32_t EXP_SECTION_BYTES = is_bfp8 ? (NUM_FACES_H * NUM_FACES_W * FACE_H) : 0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    uint32_t tiled_shape[rank], tile_strides[rank], dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        tiled_shape[i] = get_arg_val<uint32_t>(i + 3);
        tile_strides[i] = get_arg_val<uint32_t>(i + rank + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + rank * 2 + 3);
    }

    const bool is_vflip = static_cast<bool>(dims_to_flip[rank - 2]);
    const bool is_hflip = static_cast<bool>(dims_to_flip[rank - 1]);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const auto s0 = TensorAccessor(src_args, src_addr, get_tile_size(cb_id));

    auto face_data_off = [](uint32_t fi) -> uint32_t { return EXP_SECTION_BYTES + fi * FACE_HW_BYTES; };
    auto face_exp_off = [](uint32_t fi) -> uint32_t { return fi * FACE_H; };

    auto swap_bytes = [](uint8_t* a, uint8_t* b, uint32_t len) {
        for (uint32_t i = 0; i < len; i++) { uint8_t t = a[i]; a[i] = b[i]; b[i] = t; }
    };

    auto reverse_line = [&](uint8_t* p) {
        for (uint32_t lo = 0, hi = FACE_W - 1; lo < hi; lo++, hi--) {
            for (uint32_t b = 0; b < element_size; b++) {
                uint8_t t = p[lo * element_size + b];
                p[lo * element_size + b] = p[hi * element_size + b];
                p[hi * element_size + b] = t;
            }
        }
    };

    uint32_t out_md[rank];
    uint32_t rem = start_tile;
    for (uint32_t i = 0; i < rank; i++) {
        out_md[i] = rem / tile_strides[i];
        rem %= tile_strides[i];
    }

    for (uint32_t tile_id = start_tile; tile_id < end_tile; tile_id++) {
        uint32_t src_tile_id = 0;
        for (uint32_t i = 0; i < rank; i++) {
            uint32_t src_i = dims_to_flip[i] ? (tiled_shape[i] - 1 - out_md[i]) : out_md[i];
            src_tile_id += src_i * tile_strides[i];
        }

        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read_page(src_tile_id, s0, l1_addr);
        noc_async_read_barrier();

        if (is_vflip || is_hflip) {
            uint8_t* tp = reinterpret_cast<uint8_t*>(l1_addr);
            auto swap_face_pair = [&](uint32_t fa, uint32_t fb) {
                swap_bytes(tp + face_data_off(fa), tp + face_data_off(fb), FACE_HW_BYTES);
                if constexpr (is_bfp8)
                    swap_bytes(tp + face_exp_off(fa), tp + face_exp_off(fb), FACE_H);
            };

            if (is_hflip && is_vflip) { swap_face_pair(0, 3); swap_face_pair(1, 2); }
            else if (is_hflip) { swap_face_pair(0, 1); swap_face_pair(2, 3); }
            else { swap_face_pair(0, 2); swap_face_pair(1, 3); }

            const uint32_t out_tile_h = out_md[rank - 2];
            const uint32_t out_tile_w = out_md[rank - 1];

            for (uint32_t fh = 0; fh < NUM_FACES_H; fh++) {
                for (uint32_t fw = 0; fw < NUM_FACES_W; fw++) {
                    uint32_t fi = fh * NUM_FACES_W + fw;
                    uint8_t* fp = tp + face_data_off(fi);
                    uint8_t* ep = tp + face_exp_off(fi);

                    const uint32_t face_row0 = out_tile_h * TILE_H + fh * FACE_H;
                    const uint32_t face_col0 = out_tile_w * TILE_W + fw * FACE_W;

                    if (is_vflip) {
                        for (uint32_t r = 0; r < FACE_H / 2; r++) {
                            bool lo_real = (face_row0 + r) < logical_H;
                            bool hi_real = (face_row0 + FACE_H - 1 - r) < logical_H;
                            if (lo_real && hi_real) {
                                swap_bytes(fp + r * SUBTILE_LINE_BYTES,
                                           fp + (FACE_H - 1 - r) * SUBTILE_LINE_BYTES,
                                           SUBTILE_LINE_BYTES);
                                if constexpr (is_bfp8) {
                                    uint8_t t = ep[r]; ep[r] = ep[FACE_H-1-r]; ep[FACE_H-1-r] = t;
                                }
                            } else if (lo_real) {
                                for (uint32_t b = 0; b < SUBTILE_LINE_BYTES; b++)
                                    fp[r * SUBTILE_LINE_BYTES + b] = 0;
                                if constexpr (is_bfp8) ep[r] = 0;
                            }
                        }
                        for (uint32_t r = 0; r < FACE_H; r++) {
                            if (face_row0 + r >= logical_H) {
                                for (uint32_t b = 0; b < SUBTILE_LINE_BYTES; b++)
                                    fp[r * SUBTILE_LINE_BYTES + b] = 0;
                                if constexpr (is_bfp8) ep[r] = 0;
                            }
                        }
                    }

                    if (is_hflip) {
                        for (uint32_t r = 0; r < FACE_H; r++) {
                            if (face_row0 + r >= logical_H) continue;
                            if (face_col0 < logical_W)
                                reverse_line(fp + r * SUBTILE_LINE_BYTES);
                        }
                    }
                }
            }
        }

        cb_push_back(cb_id, 1);

        for (int j = (int)rank - 1; j >= 0; j--) {
            if (++out_md[j] < tiled_shape[j]) break;
            out_md[j] = 0;
        }
    }
}
