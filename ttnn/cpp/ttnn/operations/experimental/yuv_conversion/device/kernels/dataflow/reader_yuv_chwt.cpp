// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for CHWT bfloat16 input → YUV conversion (degenerate-tile approach).
//
// Pushes individual T-chunks (32 bf16 values = 64 bytes) to three CBs.
// Compute kernel processes one chunk triplet (R, G, B) → one uint8 chunk.
//
// Y pass:  reads sticks sequentially for all H×W positions.
// UV pass: reads 4 corner sticks per UV position, averages element-wise in
//          RISC-V float arithmetic, pushes pre-averaged chunk.
//          Runs twice (for Cb and Cr) to let compute apply different coefficients.
//
// Compile-time args:
//   [0] cb_R, [1] cb_G, [2] cb_B
//   [3] num_full_chunks, [4] has_partial, [5] full_chunk_bytes, [6] partial_bytes
//   [7] H, [8] W, [9] T, [10] H2, [11] W2
//   [12..] TensorAccessorArgs for input
// Runtime args: [0] src_addr

#include "api/dataflow/dataflow_api.h"

static_assert(sizeof(float) == 4);

inline float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)((u + rounding_bias) >> 16);
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_R = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B = get_compile_time_arg_val(2);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t has_partial = get_compile_time_arg_val(4);
    constexpr uint32_t full_chunk_bytes = get_compile_time_arg_val(5);  // 64
    constexpr uint32_t partial_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t H = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t H2 = get_compile_time_arg_val(10);
    constexpr uint32_t W2 = get_compile_time_arg_val(11);
    constexpr auto src_tensor_args = TensorAccessorArgs<12>();

    constexpr uint32_t HW = H * W;
    constexpr uint32_t HW2 = H2 * W2;
    constexpr uint32_t num_chunks = num_full_chunks + has_partial;
    constexpr uint32_t CHUNK_ELEMS = 32;  // elements per CB page

    const auto src = TensorAccessor(src_tensor_args, src_addr);
    const uint32_t cbs[3] = {cb_R, cb_G, cb_B};

    // Push one chunk from stick `row_id` at `chunk` to `cb_id`.
    // Zero-pads the CB page for partial chunks (partial_bytes < full_chunk_bytes).
    auto push_chunk = [&](uint32_t cb_id, uint32_t row_id, uint32_t chunk) {
        const bool is_partial = has_partial && (chunk == num_full_chunks);
        const uint32_t read_bytes = is_partial ? partial_bytes : full_chunk_bytes;
        const uint32_t byte_off = chunk * full_chunk_bytes;

        cb_reserve_back(cb_id, 1);
        uint32_t l1 = get_write_ptr(cb_id);

        if (is_partial) {
            // Zero-fill the page first, then overwrite with real data.
            volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(l1);
            for (uint32_t i = 0; i < full_chunk_bytes / 4; i++) {
                p[i] = 0;
            }
        }

        noc_async_read(src.get_noc_addr(row_id, byte_off), l1, read_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    };

    // -----------------------------------------------------------------------
    // Y pass: push sticks sequentially for all H×W positions.
    // Order: for each spatial position, push all T-chunks for R, then G, then B.
    // Compute kernel reads one (R,G,B) chunk triplet at a time.
    // -----------------------------------------------------------------------
    for (uint32_t spatial = 0; spatial < HW; spatial++) {
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            for (uint32_t c = 0; c < 3; c++) {
                push_chunk(cbs[c], c * HW + spatial, chunk);
            }
        }
    }

    // -----------------------------------------------------------------------
    // UV passes (×2 for Cb and Cr): push spatially averaged chunks.
    // For each UV position (h_uv, w_uv), the four corner sticks are read and
    // averaged element-wise in RISC-V float arithmetic.
    // -----------------------------------------------------------------------
    for (uint32_t uv_pass = 0; uv_pass < 2; uv_pass++) {
        for (uint32_t uv_idx = 0; uv_idx < HW2; uv_idx++) {
            uint32_t h_uv = uv_idx / W2, w_uv = uv_idx % W2;
            uint32_t h0 = 2 * h_uv, h1 = h0 + 1;
            uint32_t w0 = 2 * w_uv, w1 = w0 + 1;

            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                const bool is_partial = has_partial && (chunk == num_full_chunks);
                const uint32_t read_bytes = is_partial ? partial_bytes : full_chunk_bytes;
                const uint32_t byte_off = chunk * full_chunk_bytes;
                const uint32_t n_elems = read_bytes / 2;

                for (uint32_t c = 0; c < 3; c++) {
                    uint32_t c_base = c * HW;
                    uint32_t rows[4] = {
                        c_base + h0 * W + w0,
                        c_base + h0 * W + w1,
                        c_base + h1 * W + w0,
                        c_base + h1 * W + w1,
                    };

                    cb_reserve_back(cbs[c], 1);
                    uint32_t l1_out = get_write_ptr(cbs[c]);

                    // Accumulate 4 corners in RISC-V float, write averaged bf16.
                    float acc[32] = {};
                    for (uint32_t corner = 0; corner < 4; corner++) {
                        noc_async_read(src.get_noc_addr(rows[corner], byte_off), l1_out, read_bytes);
                        noc_async_read_barrier();
                        const uint16_t* s = reinterpret_cast<const uint16_t*>(l1_out);
                        for (uint32_t i = 0; i < n_elems; i++) {
                            acc[i] += bf16_to_float(s[i]);
                        }
                    }

                    uint16_t* out = reinterpret_cast<uint16_t*>(l1_out);
                    for (uint32_t i = 0; i < n_elems; i++) {
                        out[i] = float_to_bf16(acc[i] * 0.25f);
                    }
                    // Zero-pad remaining elements in the page.
                    for (uint32_t i = n_elems; i < full_chunk_bytes / 2; i++) {
                        out[i] = 0;
                    }

                    cb_push_back(cbs[c], 1);
                }
            }
        }
    }
}
