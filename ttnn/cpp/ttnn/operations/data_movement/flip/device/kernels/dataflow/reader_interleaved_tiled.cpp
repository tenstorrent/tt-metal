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
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t NUM_FACES_H = TILE_H / FACE_H;
    constexpr uint32_t NUM_FACES_W = TILE_W / FACE_W;
    constexpr uint32_t FACE_HW_BYTES = FACE_H * FACE_W * element_size;
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_W * element_size;

    constexpr bool is_bfp8 = get_named_compile_time_arg_val("is_bfp8") == 1;
    constexpr uint32_t EXP_SECTION_BYTES = is_bfp8 ? (NUM_FACES_H * NUM_FACES_W * FACE_H) : 0;
    constexpr uint32_t TILE_SIZE_BYTES = is_bfp8 ? 1088 : (TILE_H * TILE_W * element_size);

    auto face_data_off = [&](uint32_t fi) { return EXP_SECTION_BYTES + fi * FACE_HW_BYTES; };
    auto face_exp_off = [&](uint32_t fi) { return fi * FACE_H; };

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    uint32_t tiled_shape[rank], tile_strides[rank], dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        tiled_shape[i] = get_arg_val<uint32_t>(i + 3);
        tile_strides[i] = get_arg_val<uint32_t>(i + rank + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + rank + rank + 3);
    }

    const bool is_vflip = static_cast<bool>(dims_to_flip[rank - 2]);
    const bool is_hflip = static_cast<bool>(dims_to_flip[rank - 1]);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t tile_size = get_tile_size(cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr, tile_size);

    const uint32_t tiled_H = tiled_shape[rank - 2];
    const uint32_t tiled_W = tiled_shape[rank - 1];

    auto make_tile_id = [&](const uint32_t* md) -> uint32_t {
        uint32_t id = 0;
        for (uint32_t i = 0; i < rank; i++) {
            id += md[i] * tile_strides[i];
        }
        return id;
    };

    auto zero_line = [](uint8_t* p) {
        for (uint32_t b = 0; b < SUBTILE_LINE_BYTES; b++) {
            p[b] = 0;
        }
    };

    auto swap_bytes = [](uint8_t* a, uint8_t* b, uint32_t len_bytes) {
        uint32_t* wa = reinterpret_cast<uint32_t*>(a);
        uint32_t* wb = reinterpret_cast<uint32_t*>(b);
        for (uint32_t i = 0; i < len_bytes / sizeof(uint32_t); i++) {
            uint32_t t = wa[i]; wa[i] = wb[i]; wb[i] = t;
        }
    };

    auto reverse_line = [&](uint8_t* p) {
        if constexpr (element_size == 2) {
            uint16_t* q = reinterpret_cast<uint16_t*>(p);
            uint32_t lo = 0, hi = FACE_W - 1;
            while (lo < hi) { uint16_t t = q[lo]; q[lo++] = q[hi]; q[hi--] = t; }
        } else if constexpr (element_size == 4) {
            uint32_t* q = reinterpret_cast<uint32_t*>(p);
            uint32_t lo = 0, hi = FACE_W - 1;
            while (lo < hi) { uint32_t t = q[lo]; q[lo++] = q[hi]; q[hi--] = t; }
        }
    };

    uint32_t out_md[rank];
    uint32_t rem = start_tile;
    for (uint32_t i = 0; i < rank; i++) {
        out_md[i] = rem / tile_strides[i];
        rem %= tile_strides[i];
    }

    for (uint32_t tile_id = start_tile; tile_id < end_tile; tile_id++) {
        const uint32_t out_tile_h = out_md[rank - 2];
        const uint32_t out_tile_w = out_md[rank - 1];

        uint32_t src_md[rank];
        for (uint32_t i = 0; i < rank - 2; i++) {
            src_md[i] = dims_to_flip[i] ? (tiled_shape[i] - out_md[i] - 1) : out_md[i];
        }

        const uint32_t lo_r_first = out_tile_h * TILE_H;
        const uint32_t lo_r_last = lo_r_first + TILE_H - 1;
        const uint32_t lo_c_first = out_tile_w * TILE_W;
        const uint32_t lo_c_last = lo_c_first + TILE_W - 1;

        bool v_single_src_tile = true;
        if (is_vflip) {
            if (lo_r_last >= logical_H) {
                v_single_src_tile = false;
            } else {
                v_single_src_tile = ((logical_H - 1 - lo_r_first) / TILE_H == (logical_H - 1 - lo_r_last) / TILE_H);
            }
        }

        bool h_single_src_tile = true;
        if (is_hflip) {
            if (lo_c_last >= logical_W) {
                h_single_src_tile = false;
            } else {
                h_single_src_tile = ((logical_W - 1 - lo_c_first) / TILE_W == (logical_W - 1 - lo_c_last) / TILE_W);
            }
        }

        if (v_single_src_tile && h_single_src_tile) {
            src_md[rank - 2] = is_vflip ? (tiled_H - 1 - out_tile_h) : out_tile_h;
            src_md[rank - 1] = is_hflip ? (tiled_W - 1 - out_tile_w) : out_tile_w;

            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);
            noc_async_read_tile(make_tile_id(src_md), s0, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id, 1);

            uint8_t* tp = reinterpret_cast<uint8_t*>(l1_addr);

            if (is_bfp8) {
                if (is_hflip && !is_vflip) {
                    for (uint32_t fh = 0; fh < NUM_FACES_H; fh++) {
                        uint16_t* left  = (uint16_t*)(tp + face_data_off(fh * NUM_FACES_W + 0));
                        uint16_t* right = (uint16_t*)(tp + face_data_off(fh * NUM_FACES_W + 1));
                        for (uint32_t row = 0; row < FACE_H; row++) {
                            uint16_t* lr = left  + row * FACE_W;
                            uint16_t* rr = right + row * FACE_W;
                            for (uint32_t col = 0; col < FACE_W; col++) {
                                uint16_t t = lr[col];
                                lr[col] = rr[FACE_W - 1 - col];
                                rr[FACE_W - 1 - col] = t;
                            }
                        }
                    }
                } else if (is_vflip && !is_hflip) {
                    for (uint32_t fw = 0; fw < NUM_FACES_W; fw++) {
                        uint16_t* top    = (uint16_t*)(tp + face_data_off(0 * NUM_FACES_W + fw));
                        uint16_t* bottom = (uint16_t*)(tp + face_data_off(1 * NUM_FACES_W + fw));
                        for (uint32_t row = 0; row < FACE_H; row++) {
                            uint16_t* tr = top    + row * FACE_W;
                            uint16_t* br = bottom + (FACE_H - 1 - row) * FACE_W;
                            for (uint32_t col = 0; col < FACE_W; col++) {
                                uint16_t t = tr[col]; tr[col] = br[col]; br[col] = t;
                            }
                        }
                    }
                } else if (is_hflip && is_vflip) {
                    if constexpr (element_size == 2) {
                        for (uint32_t fh = 0; fh < NUM_FACES_H; fh++) {
                            for (uint32_t fw = 0; fw < NUM_FACES_W; fw++) {
                                uint32_t fi_a = fh * NUM_FACES_W + fw;
                                uint32_t fi_b = (NUM_FACES_H - 1 - fh) * NUM_FACES_W + (NUM_FACES_W - 1 - fw);
                                if (fi_a >= fi_b) continue;
                                uint16_t* fa = (uint16_t*)(tp + face_data_off(fi_a));
                                uint16_t* fb = (uint16_t*)(tp + face_data_off(fi_b));
                                for (uint32_t row = 0; row < FACE_H; row++) {
                                    uint16_t* ra = fa + row * FACE_W;
                                    uint16_t* rb = fb + (FACE_H - 1 - row) * FACE_W;
                                    for (uint32_t col = 0; col < FACE_W; col++) {
                                        uint16_t t = ra[col];
                                        ra[col] = rb[FACE_W - 1 - col];
                                        rb[FACE_W - 1 - col] = t;
                                    }
                                }
                            }
                        }
                    }
                    else if constexpr (element_size == 4) {
                        for (uint32_t fh = 0; fh < NUM_FACES_H; fh++) {
                            for (uint32_t fw = 0; fw < NUM_FACES_W; fw++) {
                                uint32_t fi_a = fh * NUM_FACES_W + fw;
                                uint32_t fi_b = (NUM_FACES_H - 1 - fh) * NUM_FACES_W + (NUM_FACES_W - 1 - fw);
                                if (fi_a >= fi_b) continue;
                                uint32_t* fa = (uint32_t*)(tp + face_data_off(fi_a));
                                uint32_t* fb = (uint32_t*)(tp + face_data_off(fi_b));
                                for (uint32_t row = 0; row < FACE_H; row++) {
                                    uint32_t* ra = fa + row * FACE_W;
                                    uint32_t* rb = fb + (FACE_H - 1 - row) * FACE_W;
                                    for (uint32_t col = 0; col < FACE_W; col++) {
                                        uint32_t t = ra[col];
                                        ra[col] = rb[FACE_W - 1 - col];
                                        rb[FACE_W - 1 - col] = t;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (is_hflip && is_vflip) {
                        swap_bytes(tp, tp + 3 * FACE_HW_BYTES, FACE_HW_BYTES);
                        swap_bytes(tp + FACE_HW_BYTES, tp + 2 * FACE_HW_BYTES, FACE_HW_BYTES);
                    } else if (is_hflip) {
                        swap_bytes(tp, tp + FACE_HW_BYTES, FACE_HW_BYTES);
                        swap_bytes(tp + 2 * FACE_HW_BYTES, tp + 3 * FACE_HW_BYTES, FACE_HW_BYTES);
                    } else if (is_vflip) {
                        swap_bytes(tp, tp + 2 * FACE_HW_BYTES, FACE_HW_BYTES);
                        swap_bytes(tp + FACE_HW_BYTES, tp + 3 * FACE_HW_BYTES, FACE_HW_BYTES);
                    }

                }

                for (uint32_t fi = 0; fi < NUM_FACES_H * NUM_FACES_W; fi++) {
                    uint8_t* fp = tp + fi * FACE_HW_BYTES;
                    if (is_vflip) {
                        for (uint32_t row = 0; row < FACE_H / 2; row++) {
                            swap_bytes(
                                fp + row * SUBTILE_LINE_BYTES,
                                fp + (FACE_H - 1 - row) * SUBTILE_LINE_BYTES,
                                SUBTILE_LINE_BYTES);
                        }
                    }
                    if (is_hflip) {
                        for (uint32_t row = 0; row < FACE_H; row++) {
                            reverse_line(fp + row * SUBTILE_LINE_BYTES);
                        }
                    }
                }
            }

            cb_push_back(cb_id, 1);

        } else {
            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);
            uint8_t* out_tp = reinterpret_cast<uint8_t*>(l1_addr);

            for (uint32_t r = 0; r < TILE_H; r++) {
                uint32_t lo_r = out_tile_h * TILE_H + r;
                uint32_t out_face_h = r / FACE_H;
                uint32_t out_row_inf = r % FACE_H;

                if (lo_r >= logical_H) {
                    for (uint32_t fc = 0; fc < NUM_FACES_W; fc++) {
                        uint32_t off = face_data_off(out_face_h * NUM_FACES_W + fc) + out_row_inf * SUBTILE_LINE_BYTES;
                        zero_line(out_tp + off);
                    }
                    continue;
                }

                uint32_t src_lo_r = is_vflip ? (logical_H - 1 - lo_r) : lo_r;
                uint32_t src_tile_h = src_lo_r / TILE_H;
                uint32_t src_row_in_tile = src_lo_r % TILE_H;
                uint32_t src_face_h = src_row_in_tile / FACE_H;
                uint32_t src_row_inf = src_row_in_tile % FACE_H;

                for (uint32_t fc = 0; fc < NUM_FACES_W; fc++) {
                    uint32_t lo_c_seg = out_tile_w * TILE_W + fc * FACE_W;
                    uint32_t out_off = face_data_off(out_face_h * NUM_FACES_W + fc) + out_row_inf * SUBTILE_LINE_BYTES;

                    if (lo_c_seg >= logical_W) {
                        zero_line(out_tp + out_off);
                        continue;
                    }

                    if (!is_hflip) {
                        uint32_t src_tile_w = lo_c_seg / TILE_W;
                        uint32_t src_fc = (lo_c_seg % TILE_W) / FACE_W;
                        uint32_t src_byte_off =
                            face_data_off(src_face_h * NUM_FACES_W + src_fc) + src_row_inf * SUBTILE_LINE_BYTES;
                        src_md[rank - 2] = src_tile_h;
                        src_md[rank - 1] = src_tile_w;
                        noc_async_read(
                            s0.get_noc_addr(make_tile_id(src_md), src_byte_off), l1_addr + out_off, SUBTILE_LINE_BYTES);

                        if (lo_c_seg + FACE_W > logical_W) {
                            uint32_t real_cols = logical_W - lo_c_seg;
                            uint8_t* pad = out_tp + out_off + real_cols * element_size;
                            for (uint32_t b = 0; b < (FACE_W - real_cols) * element_size; b++) {
                                pad[b] = 0;
                            }
                        }
                    } else {
                        uint32_t src_lo_c_last = logical_W - 1 - lo_c_seg;
                        uint32_t n_real = (lo_c_seg + FACE_W <= logical_W) ? FACE_W : (logical_W - lo_c_seg);
                        uint32_t src_lo_c_first = src_lo_c_last - n_real + 1;
                        uint32_t src_tile_w = src_lo_c_last / TILE_W;
                        uint32_t src_tile_w_first = src_lo_c_first / TILE_W;
                        uint32_t src_fc = (src_lo_c_last % TILE_W) / FACE_W;
                        src_md[rank - 2] = src_tile_h;
                        src_md[rank - 1] = src_tile_w;
                        uint32_t stid_hi = make_tile_id(src_md);

                        if (src_tile_w == src_tile_w_first) {
                            uint32_t src_fc_first = (src_lo_c_first % TILE_W) / FACE_W;
                            if (src_fc == src_fc_first) {
                                uint32_t col_end = src_lo_c_last % TILE_W % FACE_W;
                                uint32_t col_start = col_end - n_real + 1;
                                uint32_t off = face_data_off(src_face_h * NUM_FACES_W + src_fc) +
                                               src_row_inf * SUBTILE_LINE_BYTES + col_start * element_size;
                                noc_async_read(s0.get_noc_addr(stid_hi, off), l1_addr + out_off, n_real * element_size);
                                noc_async_read_barrier();
                            } else {
                                uint32_t n_first = src_fc * FACE_W - (src_lo_c_first % TILE_W);
                                uint32_t n_hi = n_real - n_first;
                                uint32_t off_first = face_data_off(src_face_h * NUM_FACES_W + src_fc_first) +
                                                     src_row_inf * SUBTILE_LINE_BYTES +
                                                     ((src_lo_c_first % TILE_W) % FACE_W) * element_size;
                                noc_async_read(
                                    s0.get_noc_addr(stid_hi, off_first), l1_addr + out_off, n_first * element_size);
                                uint32_t off_hi =
                                    face_data_off(src_face_h * NUM_FACES_W + src_fc) + src_row_inf * SUBTILE_LINE_BYTES;
                                noc_async_read(
                                    s0.get_noc_addr(stid_hi, off_hi),
                                    l1_addr + out_off + n_first * element_size,
                                    n_hi * element_size);
                                noc_async_read_barrier();
                            }
                        } else {
                            // Source columns span two different tiles.
                            uint32_t n_lo = src_tile_w * TILE_W - src_lo_c_first;
                            uint32_t n_hi = n_real - n_lo;
                            src_md[rank - 1] = src_tile_w_first;
                            uint32_t stid_lo = make_tile_id(src_md);
                            uint32_t fc_lo = (src_lo_c_first % TILE_W) / FACE_W;
                            uint32_t off_lo = face_data_off(src_face_h * NUM_FACES_W + fc_lo) +
                                              src_row_inf * SUBTILE_LINE_BYTES +
                                              ((src_lo_c_first % TILE_W) % FACE_W) * element_size;
                            noc_async_read(s0.get_noc_addr(stid_lo, off_lo), l1_addr + out_off, n_lo * element_size);
                            uint32_t off_hi =
                                face_data_off(src_face_h * NUM_FACES_W + src_fc) + src_row_inf * SUBTILE_LINE_BYTES;
                            noc_async_read(
                                s0.get_noc_addr(stid_hi, off_hi),
                                l1_addr + out_off + n_lo * element_size,
                                n_hi * element_size);
                            noc_async_read_barrier();
                        }

                        uint8_t* seg = out_tp + out_off;
                        uint32_t lo2 = 0, hi2 = n_real - 1;
                        while (lo2 < hi2) {
                            for (uint32_t b = 0; b < element_size; b++) {
                                uint8_t t = seg[lo2 * element_size + b];
                                seg[lo2 * element_size + b] = seg[hi2 * element_size + b];
                                seg[hi2 * element_size + b] = t;
                            }
                            lo2++;
                            hi2--;
                        }

                        if (n_real < FACE_W) {
                            uint8_t* pad = out_tp + out_off + n_real * element_size;
                            for (uint32_t b = 0; b < (FACE_W - n_real) * element_size; b++) {
                                pad[b] = 0;
                            }
                        }
                    }
                }
            }

            if (!is_hflip) {
                noc_async_read_barrier();
            }

            if (is_bfp8) {
                for (uint32_t r = 0; r < TILE_H; r++) {
                    uint32_t lo_r = out_tile_h * TILE_H + r;
                    uint32_t out_fi_base = (r / FACE_H) * NUM_FACES_W;
                    uint32_t out_row_inf = r % FACE_H;

                    if (lo_r >= logical_H) {
                        for (uint32_t fc = 0; fc < NUM_FACES_W; fc++) {
                            out_tp[face_exp_off(out_fi_base + fc) + out_row_inf] = 0;
                        }
                        continue;
                    }

                    uint32_t src_lo_r = is_vflip ? (logical_H - 1 - lo_r) : lo_r;
                    uint32_t src_tile_h = src_lo_r / TILE_H;
                    uint32_t src_face_h = (src_lo_r % TILE_H) / FACE_H;
                    uint32_t src_row_inf = src_lo_r % FACE_H;

                    for (uint32_t fc = 0; fc < NUM_FACES_W; fc++) {
                        uint32_t lo_c_seg = out_tile_w * TILE_W + fc * FACE_W;

                        if (lo_c_seg >= logical_W) {
                            out_tp[face_exp_off(out_fi_base + fc) + out_row_inf] = 0;
                            continue;
                        }

                        uint32_t src_lo_c = is_hflip ? (logical_W - 1 - lo_c_seg) : lo_c_seg;
                        uint32_t src_tile_w = src_lo_c / TILE_W;
                        uint32_t src_fc = (src_lo_c % TILE_W) / FACE_W;
                        uint32_t src_fi = src_face_h * NUM_FACES_W + src_fc;
                        src_md[rank - 2] = src_tile_h;
                        src_md[rank - 1] = src_tile_w;
                        noc_async_read(
                            s0.get_noc_addr(make_tile_id(src_md), face_exp_off(src_fi) + src_row_inf),
                            l1_addr + face_exp_off(out_fi_base + fc) + out_row_inf,
                            1);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_id, 1);
        }

        for (int j = (int)rank - 1; j >= 0; j--) {
            if (++out_md[j] < tiled_shape[j]) {
                break;
            }
            out_md[j] = 0;
        }
    }
}
