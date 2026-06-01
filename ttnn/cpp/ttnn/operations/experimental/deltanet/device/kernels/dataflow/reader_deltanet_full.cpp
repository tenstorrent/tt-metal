// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for fully fused DeltaNet decode (phase B2).
//
// Performs conv1d + silu on raw qkv_proj using conv_state and conv1d_weight,
// producing q/k/v directly. Also reads b_proj, a_proj, z_proj, state, and
// static weights. Computes decay/beta scalars from b/a projections.

#include <cstdint>
#include <cmath>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

FORCE_INLINE uint16_t f32_to_bf16(float f32) {
    uint32_t f32_bits;
    __builtin_memcpy(&f32_bits, &f32, sizeof(uint32_t));
    return static_cast<uint16_t>((f32_bits + 0x8000) >> 16);
}

// Extract element [0, idx] from a bf16 TILE-layout tile (1D vector tile)
FORCE_INLINE float extract_bf16_element_1d(uint32_t tile_l1_addr, uint32_t idx) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    uint32_t face = idx / 16;
    uint32_t pos = face * 256 + (idx % 16);
    return bf16_to_f32(tile[pos]);
}

// Write element [0, idx] into a bf16 TILE-layout tile (1D vector tile)
FORCE_INLINE void write_bf16_element_1d(uint32_t tile_l1_addr, uint32_t idx, uint16_t val) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    uint32_t face = idx / 16;
    uint32_t pos = face * 256 + (idx % 16);
    tile[pos] = val;
}

// For conv_state/weight tiles [conv_dim, 32]: element at (row, col)
// row = channel % 32, col = kernel position (0..3)
// Face layout: rows 0-15 in faces 0,1; rows 16-31 in faces 2,3
FORCE_INLINE uint32_t conv_tile_offset(uint32_t row, uint32_t col) {
    uint32_t face = (row >= 16) ? 2 : 0;
    return face * 256 + (row % 16) * 16 + col;
}

// Write a scalar as bf16 broadcast tile (scalar in position [0,0] of all faces)
FORCE_INLINE void write_bf16_scalar_tile(uint32_t tile_l1_addr, float value) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    for (uint32_t i = 0; i < 1024; i++) {
        tile[i] = 0;
    }
    uint16_t val_bf16 = f32_to_bf16(value);
    tile[0] = val_bf16;
    tile[256] = val_bf16;
    tile[512] = val_bf16;
    tile[768] = val_bf16;
}

FORCE_INLINE float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

void kernel_main() {
    constexpr uint32_t cb_state       = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q           = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k           = get_compile_time_arg_val(2);
    constexpr uint32_t cb_v           = get_compile_time_arg_val(3);
    constexpr uint32_t cb_g           = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta        = get_compile_time_arg_val(5);
    constexpr uint32_t cb_z           = get_compile_time_arg_val(6);
    constexpr uint32_t cb_norm_w      = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_T         = get_compile_time_arg_val(8);
    constexpr uint32_t Dk_tiles       = get_compile_time_arg_val(9);
    constexpr uint32_t Dv_tiles       = get_compile_time_arg_val(10);
    constexpr uint32_t H              = get_compile_time_arg_val(11);
    constexpr uint32_t Hk             = get_compile_time_arg_val(12);
    constexpr uint32_t Dk             = get_compile_time_arg_val(13);
    constexpr uint32_t Dv             = get_compile_time_arg_val(14);
    constexpr uint32_t conv_dim       = get_compile_time_arg_val(15);
    constexpr uint32_t conv_k         = get_compile_time_arg_val(16);
    constexpr uint32_t head_expand    = get_compile_time_arg_val(17);
    constexpr uint32_t cb_conv_scratch    = get_compile_time_arg_val(18);
    constexpr uint32_t cb_conv_state_out  = get_compile_time_arg_val(19);
    constexpr uint32_t cb_scaler          = get_compile_time_arg_val(20);
    constexpr uint32_t cb_eps             = get_compile_time_arg_val(21);
    constexpr auto accessor_args      = TensorAccessorArgs<22>();

    const uint32_t state_addr         = get_arg_val<uint32_t>(0);
    const uint32_t qkv_addr           = get_arg_val<uint32_t>(1);
    const uint32_t z_addr             = get_arg_val<uint32_t>(2);
    const uint32_t b_addr             = get_arg_val<uint32_t>(3);
    const uint32_t a_addr             = get_arg_val<uint32_t>(4);
    const uint32_t conv_state_addr    = get_arg_val<uint32_t>(5);
    const uint32_t conv_w_addr        = get_arg_val<uint32_t>(6);
    const uint32_t a_log_addr         = get_arg_val<uint32_t>(7);
    const uint32_t dt_bias_addr       = get_arg_val<uint32_t>(8);
    const uint32_t norm_w_addr        = get_arg_val<uint32_t>(9);
    const uint32_t state_start_tile   = get_arg_val<uint32_t>(10);
    const uint32_t head_idx           = get_arg_val<uint32_t>(11);
    const uint32_t q_byte_offset      = get_arg_val<uint32_t>(12);
    const uint32_t k_byte_offset      = get_arg_val<uint32_t>(13);
    const uint32_t v_byte_offset      = get_arg_val<uint32_t>(14);
    const uint32_t z_byte_offset      = get_arg_val<uint32_t>(15);
    const uint32_t conv_q_state_tile  = get_arg_val<uint32_t>(16);
    const uint32_t conv_k_state_tile  = get_arg_val<uint32_t>(17);
    const uint32_t conv_v_state_tile  = get_arg_val<uint32_t>(18);
    const uint32_t qkv_q_tile        = get_arg_val<uint32_t>(19);
    const uint32_t qkv_k_tile        = get_arg_val<uint32_t>(20);
    const uint32_t qkv_v_tile        = get_arg_val<uint32_t>(21);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    constexpr uint32_t TILES_PER_COMPONENT = 4;  // 128 channels / 32

    const uint32_t tile_bytes = get_tile_size(cb_q);  // bf16 = 2048

    const auto state_acc     = TensorAccessor(accessor_args, state_addr, tile_bytes);
    const auto qkv_acc       = TensorAccessor(accessor_args, qkv_addr, tile_bytes);
    const auto z_acc         = TensorAccessor(accessor_args, z_addr, tile_bytes);
    const auto b_acc         = TensorAccessor(accessor_args, b_addr, tile_bytes);
    const auto a_acc         = TensorAccessor(accessor_args, a_addr, tile_bytes);
    const auto conv_state_acc = TensorAccessor(accessor_args, conv_state_addr, tile_bytes);
    const auto conv_w_acc    = TensorAccessor(accessor_args, conv_w_addr, tile_bytes);
    const auto a_log_acc     = TensorAccessor(accessor_args, a_log_addr, tile_bytes);
    const auto dt_acc        = TensorAccessor(accessor_args, dt_bias_addr, tile_bytes);
    const auto norm_acc      = TensorAccessor(accessor_args, norm_w_addr, tile_bytes);

    // -----------------------------------------------------------------------
    // 1. Read recurrent state [state_tiles tiles, bf16]
    // -----------------------------------------------------------------------
    {
        cb_reserve_back(cb_state, state_tiles);
        uint32_t l1_addr = get_write_ptr(cb_state);
        for (uint32_t t = 0; t < state_tiles; t++) {
            noc_async_read_tile(state_start_tile + t, state_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_state, state_tiles);
    }

    // -----------------------------------------------------------------------
    // 2. Conv1d + SiLU → produce q, k, v in CBs
    //    Process each component sequentially to limit L1 scratch to 12 tiles.
    //    conv_state: [1,1,conv_dim,32] TILE — channel c at row c, cols 0..3
    //    conv1d_weight: same layout
    //    qkv_proj: [1,1,1,conv_dim] TILE — element c in tile c/32
    // -----------------------------------------------------------------------
    {
        // Scratch area: read conv_state(4) + conv_weight(4) + qkv(4) tiles
        cb_reserve_back(cb_conv_scratch, 12);
        uint32_t scratch_base = get_write_ptr(cb_conv_scratch);

        // Also reserve conv_state_out for all 12 tiles (q+k+v updated state)
        cb_reserve_back(cb_conv_state_out, 12);
        uint32_t conv_out_base = get_write_ptr(cb_conv_state_out);

        // Component tile starts: [conv_state_start, qkv_start, output_cb, num_tiles]
        uint32_t comp_conv_tiles[3] = {conv_q_state_tile, conv_k_state_tile, conv_v_state_tile};
        uint32_t comp_qkv_tiles[3]  = {qkv_q_tile, qkv_k_tile, qkv_v_tile};
        uint32_t comp_cb[3]         = {cb_q, cb_k, cb_v};

        for (uint32_t comp = 0; comp < 3; comp++) {
            uint32_t cs_start = comp_conv_tiles[comp];
            uint32_t qkv_start = comp_qkv_tiles[comp];

            // Read 4 conv_state tiles into scratch [0..3]
            uint32_t cs_l1 = scratch_base;
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_read_tile(cs_start + t, conv_state_acc, cs_l1);
                cs_l1 += tile_bytes;
            }

            // Read 4 conv_weight tiles into scratch [4..7]
            uint32_t cw_l1 = scratch_base + 4 * tile_bytes;
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_read_tile(cs_start + t, conv_w_acc, cw_l1);
                cw_l1 += tile_bytes;
            }

            // Read 4 qkv_proj tiles into scratch [8..11]
            uint32_t qkv_l1 = scratch_base + 8 * tile_bytes;
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_read_tile(qkv_start + t, qkv_acc, qkv_l1);
                qkv_l1 += tile_bytes;
            }

            noc_async_read_barrier();

            // Reserve output CB for this component
            cb_reserve_back(comp_cb[comp], TILES_PER_COMPONENT);
            uint32_t out_l1 = get_write_ptr(comp_cb[comp]);

            // Zero output tiles
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1 + t * tile_bytes);
                for (uint32_t i = 0; i < 1024; i++) { dst[i] = 0; }
            }

            // Process each tile (32 channels each)
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                uint32_t cs_tile_addr = scratch_base + t * tile_bytes;
                uint32_t cw_tile_addr = scratch_base + (4 + t) * tile_bytes;
                uint32_t qkv_tile_addr = scratch_base + (8 + t) * tile_bytes;
                uint32_t out_tile_addr = out_l1 + t * tile_bytes;

                volatile tt_l1_ptr uint16_t* cs_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs_tile_addr);
                volatile tt_l1_ptr uint16_t* cw_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cw_tile_addr);

                for (uint32_t c = 0; c < 32; c++) {
                    // Extract conv_state[c, 0..3] and weight[c, 0..3]
                    uint32_t off0 = conv_tile_offset(c, 0);
                    float s0 = bf16_to_f32(cs_tile[off0]);
                    float s1 = bf16_to_f32(cs_tile[off0 + 1]);
                    float s2 = bf16_to_f32(cs_tile[off0 + 2]);
                    float s3 = bf16_to_f32(cs_tile[off0 + 3]);

                    float w0 = bf16_to_f32(cw_tile[off0]);
                    float w1 = bf16_to_f32(cw_tile[off0 + 1]);
                    float w2 = bf16_to_f32(cw_tile[off0 + 2]);
                    float w3 = bf16_to_f32(cw_tile[off0 + 3]);

                    // New qkv value for this channel
                    float new_val = extract_bf16_element_1d(qkv_tile_addr, c);

                    // Conv1d: roll(-1) + insert + dot product
                    // new_state = [s1, s2, s3, new_val]
                    float dot = s1 * w0 + s2 * w1 + s3 * w2 + new_val * w3;

                    // SiLU activation
                    float result = silu_f32(dot);

                    // Write result to output CB tile
                    write_bf16_element_1d(out_tile_addr, c, f32_to_bf16(result));

                    // Update conv_state tile in-place for writeback
                    cs_tile[off0]     = f32_to_bf16(s1);
                    cs_tile[off0 + 1] = f32_to_bf16(s2);
                    cs_tile[off0 + 2] = f32_to_bf16(s3);
                    cs_tile[off0 + 3] = f32_to_bf16(new_val);
                }
            }

            // L2-normalize q (comp=0) and k (comp=1) in-place before pushing
            if (comp < 2) {
                float sum_sq = 0.0f;
                for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                    uint32_t addr = out_l1 + t * tile_bytes;
                    for (uint32_t c = 0; c < 32; c++) {
                        float v = extract_bf16_element_1d(addr, c);
                        sum_sq += v * v;
                    }
                }
                constexpr float l2_eps = 1e-6f;
                float inv_norm = 1.0f / sqrtf(sum_sq + l2_eps);
                if (comp == 0) {
                    constexpr float scale = 1.0f / sqrtf(static_cast<float>(Dk));
                    inv_norm *= scale;
                }
                for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                    uint32_t addr = out_l1 + t * tile_bytes;
                    for (uint32_t c = 0; c < 32; c++) {
                        float v = extract_bf16_element_1d(addr, c);
                        write_bf16_element_1d(addr, c, f32_to_bf16(v * inv_norm));
                    }
                }
            }

            cb_push_back(comp_cb[comp], TILES_PER_COMPONENT);

            // Copy updated conv_state tiles to conv_state_out CB
            uint32_t conv_out_addr = conv_out_base + comp * TILES_PER_COMPONENT * tile_bytes;
            volatile tt_l1_ptr uint8_t* dst_bytes = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(conv_out_addr);
            volatile tt_l1_ptr uint8_t* src_bytes = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch_base);
            for (uint32_t b = 0; b < TILES_PER_COMPONENT * tile_bytes; b++) {
                dst_bytes[b] = src_bytes[b];
            }
        }

        cb_push_back(cb_conv_state_out, 12);
    }

    // -----------------------------------------------------------------------
    // 3. Construct k_T (transposed k)
    // -----------------------------------------------------------------------
    {
        cb_reserve_back(cb_k_T, Dk_tiles);
        uint32_t k_src = get_read_ptr(cb_k);
        uint32_t k_T_dst = get_write_ptr(cb_k_T);
        for (uint32_t t = 0; t < Dk_tiles; t++) {
            volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_src);
            volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_T_dst);
            for (uint32_t i = 0; i < 1024; i++) { dst[i] = 0; }
            for (uint32_t j = 0; j < 16; j++) { dst[j * 16] = src[j]; }
            for (uint32_t j = 0; j < 16; j++) { dst[512 + j * 16] = src[256 + j]; }
            k_src += tile_bytes;
            k_T_dst += tile_bytes;
        }
        cb_push_back(cb_k_T, Dk_tiles);
    }

    // -----------------------------------------------------------------------
    // 4. Read b_proj, a_proj, A_log, dt_bias → compute decay & beta scalars
    // -----------------------------------------------------------------------
    {
        uint32_t b_tile_idx = head_idx / 32;
        uint32_t b_elem_idx = head_idx % 32;
        cb_reserve_back(cb_beta, 1);
        uint32_t beta_l1 = get_write_ptr(cb_beta);
        noc_async_read_tile(b_tile_idx, b_acc, beta_l1);
        noc_async_read_barrier();
        float b_val = extract_bf16_element_1d(beta_l1, b_elem_idx);

        cb_reserve_back(cb_g, 1);
        uint32_t g_l1 = get_write_ptr(cb_g);
        uint32_t a_tile_idx = head_idx / 32;
        uint32_t a_elem_idx = head_idx % 32;
        noc_async_read_tile(a_tile_idx, a_acc, g_l1);
        noc_async_read_barrier();
        float a_val = extract_bf16_element_1d(g_l1, a_elem_idx);

        noc_async_read_tile(a_tile_idx, a_log_acc, beta_l1);
        noc_async_read_barrier();
        float a_log_val = extract_bf16_element_1d(beta_l1, a_elem_idx);

        noc_async_read_tile(a_tile_idx, dt_acc, beta_l1);
        noc_async_read_barrier();
        float dt_bias_val = extract_bf16_element_1d(beta_l1, a_elem_idx);

        float beta_val = 1.0f / (1.0f + expf(-b_val));

        float a_plus_dt = a_val + dt_bias_val;
        float sp = (a_plus_dt > 20.0f) ? a_plus_dt : logf(1.0f + expf(a_plus_dt));
        float decay_val = expf(-expf(a_log_val) * sp);

        write_bf16_scalar_tile(g_l1, decay_val);
        cb_push_back(cb_g, 1);

        write_bf16_scalar_tile(beta_l1, beta_val);
        cb_push_back(cb_beta, 1);
    }

    // -----------------------------------------------------------------------
    // 5. Read z tiles [Dv_tiles, bf16]
    // -----------------------------------------------------------------------
    {
        uint32_t z_start_tile = head_idx * Dv_tiles;
        cb_reserve_back(cb_z, Dv_tiles);
        uint32_t l1_addr = get_write_ptr(cb_z);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_read_tile(z_start_tile + t, z_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_z, Dv_tiles);
    }

    // -----------------------------------------------------------------------
    // 6. Read norm_weight [Dv_tiles, bf16]
    // -----------------------------------------------------------------------
    {
        cb_reserve_back(cb_norm_w, Dv_tiles);
        uint32_t l1_addr = get_write_ptr(cb_norm_w);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_read_tile(t, norm_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_norm_w, Dv_tiles);
    }

    // -----------------------------------------------------------------------
    // 7. Generate scaler tile for RMSNorm mean computation
    //    REDUCE_SCALAR applies scaler twice (row then col), so use 1/sqrt(Dv)
    //    to get effective 1/Dv. Row-0 fill: value in row 0 of all faces.
    // -----------------------------------------------------------------------
    {
        cb_reserve_back(cb_scaler, 1);
        uint32_t scaler_l1 = get_write_ptr(cb_scaler);
        float inv_sqrt_dv = 1.0f / sqrtf(static_cast<float>(Dv));
        volatile tt_l1_ptr uint16_t* stile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scaler_l1);
        for (uint32_t i = 0; i < 1024; i++) stile[i] = 0;
        uint16_t sv = f32_to_bf16(inv_sqrt_dv);
        for (uint32_t face = 0; face < 4; face++) {
            uint32_t base = face * 256;
            for (uint32_t c = 0; c < 16; c++) {
                stile[base + c] = sv;
            }
        }
        cb_push_back(cb_scaler, 1);
    }

    // -----------------------------------------------------------------------
    // 8. Generate epsilon tile for RMSNorm (eps = 1e-6)
    //    Only face 0 [0,0] matters (REDUCE_SCALAR result is there).
    // -----------------------------------------------------------------------
    {
        cb_reserve_back(cb_eps, 1);
        uint32_t eps_l1 = get_write_ptr(cb_eps);
        write_bf16_scalar_tile(eps_l1, 1e-6f);
        cb_push_back(cb_eps, 1);
    }
}
