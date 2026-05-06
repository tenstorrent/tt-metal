// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for CHWT bfloat16 input → YUV uint8 output (pure RISC-V, multicore).
//
// All color-space math (linear combination, clamp, uint8 cast) runs in
// scalar float arithmetic on BRISC.  No compute kernel is needed.
//
// Each core handles a contiguous slice of spatial positions for all 3 planes.
// Three sequential passes push uint8 chunks into cb_out:
//   Y  — full resolution (per-core slice)
//   Cb — half resolution with 2×2 spatial averaging (per-core slice)
//   Cr — half resolution with 2×2 spatial averaging (per-core slice)
//
// Compile-time args:
//   [0] cb_R, [1] cb_G, [2] cb_B  — staging CBs (reader-only L1 scratch)
//   [3] cb_out                     — output CB (shared with writer)
//   [4] num_full_chunks, [5] has_partial
//   [6] full_chunk_bytes, [7] partial_bytes
//   [8] H, [9] W, [10] T, [11] H2, [12] W2
//   [13..] TensorAccessorArgs for input
//
// Runtime args:
//   [0]     src_addr
//   [1..4]  Y  coefficients: w_r, w_g, w_b, offset  (float32 bit patterns)
//   [5..8]  Cb coefficients
//   [9..12] Cr coefficients
//   [13] y_start   — first Y spatial index for this core
//   [14] y_count   — number of Y spatial positions
//   [15] uv_start  — first UV spatial index for this core
//   [16] uv_count  — number of UV spatial positions

#include "api/dataflow/dataflow_api.h"

static_assert(sizeof(float) == 4);

inline float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

inline float arg_to_float(uint32_t idx) {
    uint32_t bits = get_arg_val<uint32_t>(idx);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_R = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t has_partial = get_compile_time_arg_val(5);
    constexpr uint32_t full_chunk_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t partial_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t H = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);
    constexpr uint32_t T = get_compile_time_arg_val(10);
    constexpr uint32_t H2 = get_compile_time_arg_val(11);
    constexpr uint32_t W2 = get_compile_time_arg_val(12);
    constexpr auto src_tensor_args = TensorAccessorArgs<13>();

    constexpr uint32_t HW = H * W;
    constexpr uint32_t num_chunks = num_full_chunks + has_partial;
    constexpr uint32_t CHUNK_ELEMS = 32;

    const auto src = TensorAccessor(src_tensor_args, src_addr);

    // Decode coefficients from runtime args (float32 stored as uint32 bits).
    float coeff_wr[3], coeff_wg[3], coeff_wb[3], coeff_off[3];
    for (uint32_t p = 0; p < 3; p++) {
        uint32_t base = 1 + p * 4;
        coeff_wr[p] = arg_to_float(base + 0);
        coeff_wg[p] = arg_to_float(base + 1);
        coeff_wb[p] = arg_to_float(base + 2);
        coeff_off[p] = arg_to_float(base + 3);
    }

    // Per-core work bounds.
    const uint32_t y_start = get_arg_val<uint32_t>(13);
    const uint32_t y_count = get_arg_val<uint32_t>(14);
    const uint32_t uv_start = get_arg_val<uint32_t>(15);
    const uint32_t uv_count = get_arg_val<uint32_t>(16);

    // Reserve staging CB pages once and reuse their L1 addresses throughout.
    // These CBs have no consumer — they are pure scratch for NOC reads.
    cb_reserve_back(cb_R, 1);
    uint32_t l1_r = get_write_ptr(cb_R);
    cb_reserve_back(cb_G, 1);
    uint32_t l1_g = get_write_ptr(cb_G);
    cb_reserve_back(cb_B, 1);
    uint32_t l1_b = get_write_ptr(cb_B);

    const uint32_t y_end = y_start + y_count;
    const uint32_t uv_end = uv_start + uv_count;

    // -------------------------------------------------------------------
    // Y pass: full resolution (this core's slice)
    // -------------------------------------------------------------------
    {
        const float wr = coeff_wr[0], wg = coeff_wg[0], wb = coeff_wb[0], off = coeff_off[0];

        for (uint32_t spatial = y_start; spatial < y_end; spatial++) {
            const uint32_t r_row = 0 * HW + spatial;
            const uint32_t g_row = 1 * HW + spatial;
            const uint32_t b_row = 2 * HW + spatial;

            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                const bool is_partial = has_partial && (chunk == num_full_chunks);
                const uint32_t read_bytes = is_partial ? partial_bytes : full_chunk_bytes;
                const uint32_t n_elems = read_bytes / 2;
                const uint32_t byte_off = chunk * full_chunk_bytes;

                noc_async_read(src.get_noc_addr(r_row, byte_off), l1_r, read_bytes);
                noc_async_read(src.get_noc_addr(g_row, byte_off), l1_g, read_bytes);
                noc_async_read(src.get_noc_addr(b_row, byte_off), l1_b, read_bytes);
                noc_async_read_barrier();

                const uint16_t* rp = reinterpret_cast<const uint16_t*>(l1_r);
                const uint16_t* gp = reinterpret_cast<const uint16_t*>(l1_g);
                const uint16_t* bp = reinterpret_cast<const uint16_t*>(l1_b);

                cb_reserve_back(cb_out, 1);
                uint8_t* out = reinterpret_cast<uint8_t*>(get_write_ptr(cb_out));

                for (uint32_t i = 0; i < n_elems; i++) {
                    float val = wr * bf16_to_float(rp[i]) + wg * bf16_to_float(gp[i]) + wb * bf16_to_float(bp[i]) + off;
                    if (val < 0.0f) {
                        val = 0.0f;
                    }
                    if (val > 255.0f) {
                        val = 255.0f;
                    }
                    out[i] = (uint8_t)val;
                }
                for (uint32_t i = n_elems; i < CHUNK_ELEMS; i++) {
                    out[i] = 0;
                }
                cb_push_back(cb_out, 1);
            }
        }
    }

    // -------------------------------------------------------------------
    // Cb and Cr passes: half resolution with 2×2 spatial averaging
    // -------------------------------------------------------------------
    for (uint32_t pass = 1; pass <= 2; pass++) {
        const float wr = coeff_wr[pass], wg = coeff_wg[pass];
        const float wb = coeff_wb[pass], off = coeff_off[pass];

        for (uint32_t uv_idx = uv_start; uv_idx < uv_end; uv_idx++) {
            const uint32_t h_uv = uv_idx / W2, w_uv = uv_idx % W2;
            const uint32_t h0 = 2 * h_uv, h1 = h0 + 1;
            const uint32_t w0 = 2 * w_uv, w1 = w0 + 1;

            const uint32_t corner_rows[4] = {
                h0 * W + w0,
                h0 * W + w1,
                h1 * W + w0,
                h1 * W + w1,
            };

            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                const bool is_partial = has_partial && (chunk == num_full_chunks);
                const uint32_t read_bytes = is_partial ? partial_bytes : full_chunk_bytes;
                const uint32_t n_elems = read_bytes / 2;
                const uint32_t byte_off = chunk * full_chunk_bytes;

                float avg_r[CHUNK_ELEMS] = {};
                float avg_g[CHUNK_ELEMS] = {};
                float avg_b[CHUNK_ELEMS] = {};

                for (uint32_t corner = 0; corner < 4; corner++) {
                    const uint32_t base_row = corner_rows[corner];

                    noc_async_read(src.get_noc_addr(0 * HW + base_row, byte_off), l1_r, read_bytes);
                    noc_async_read(src.get_noc_addr(1 * HW + base_row, byte_off), l1_g, read_bytes);
                    noc_async_read(src.get_noc_addr(2 * HW + base_row, byte_off), l1_b, read_bytes);
                    noc_async_read_barrier();

                    const uint16_t* rp = reinterpret_cast<const uint16_t*>(l1_r);
                    const uint16_t* gp = reinterpret_cast<const uint16_t*>(l1_g);
                    const uint16_t* bp = reinterpret_cast<const uint16_t*>(l1_b);

                    for (uint32_t i = 0; i < n_elems; i++) {
                        avg_r[i] += bf16_to_float(rp[i]);
                        avg_g[i] += bf16_to_float(gp[i]);
                        avg_b[i] += bf16_to_float(bp[i]);
                    }
                }

                cb_reserve_back(cb_out, 1);
                uint8_t* out = reinterpret_cast<uint8_t*>(get_write_ptr(cb_out));

                for (uint32_t i = 0; i < n_elems; i++) {
                    float r = avg_r[i] * 0.25f;
                    float g = avg_g[i] * 0.25f;
                    float b = avg_b[i] * 0.25f;
                    float val = wr * r + wg * g + wb * b + off;
                    if (val < 0.0f) {
                        val = 0.0f;
                    }
                    if (val > 255.0f) {
                        val = 255.0f;
                    }
                    out[i] = (uint8_t)val;
                }
                for (uint32_t i = n_elems; i < CHUNK_ELEMS; i++) {
                    out[i] = 0;
                }
                cb_push_back(cb_out, 1);
            }
        }
    }
}
