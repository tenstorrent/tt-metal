// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for DeltaNet prefill (S>1 token loop).
//
// One-time reads: recurrent state, conv_state, conv_weight.
// Per-token loop: reads qkv/z/b/a projections for token s, performs conv1d+silu,
// L2norm, k_T construction, decay/beta computation.
//
// Input projection tensors are [1, 1, S, dim] in TILE layout. Token s occupies
// tile_row = s/32, within-tile row = s%32. Tiles within the same tile_row are
// cached to avoid redundant DRAM reads.

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

// Extract element [row, col] from a bf16 TILE (32x32)
FORCE_INLINE float extract_bf16_element_2d(uint32_t tile_l1_addr, uint32_t row, uint32_t col) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    uint32_t face_row = (row >= 16) ? 2 : 0;
    uint32_t face_col = (col >= 16) ? 1 : 0;
    uint32_t face = face_row + face_col;
    uint32_t pos = face * 256 + (row % 16) * 16 + (col % 16);
    return bf16_to_f32(tile[pos]);
}

// Extract element [0, idx] from a bf16 TILE (1D vector in row 0)
FORCE_INLINE float extract_bf16_element_1d(uint32_t tile_l1_addr, uint32_t idx) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    uint32_t face = idx / 16;
    uint32_t pos = face * 256 + (idx % 16);
    return bf16_to_f32(tile[pos]);
}

// Write element [0, idx] into row 0 of a bf16 TILE
FORCE_INLINE void write_bf16_element_1d(uint32_t tile_l1_addr, uint32_t idx, uint16_t val) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    uint32_t face = idx / 16;
    uint32_t pos = face * 256 + (idx % 16);
    tile[pos] = val;
}

FORCE_INLINE uint32_t conv_tile_offset(uint32_t row, uint32_t col) {
    uint32_t face = (row >= 16) ? 2 : 0;
    return face * 256 + (row % 16) * 16 + col;
}

FORCE_INLINE void write_bf16_scalar_tile(uint32_t tile_l1_addr, float value) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    for (uint32_t i = 0; i < 1024; i++) tile[i] = 0;
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
    constexpr uint32_t S                  = get_compile_time_arg_val(22);
    constexpr uint32_t qkv_col_tiles      = get_compile_time_arg_val(23);
    constexpr uint32_t z_col_tiles        = get_compile_time_arg_val(24);
    constexpr uint32_t ba_col_tiles       = get_compile_time_arg_val(25);
    constexpr auto accessor_args      = TensorAccessorArgs<26>();

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
    const uint32_t conv_q_tile        = get_arg_val<uint32_t>(12);
    const uint32_t conv_k_tile        = get_arg_val<uint32_t>(13);
    const uint32_t conv_v_tile        = get_arg_val<uint32_t>(14);
    const uint32_t qkv_q_tile         = get_arg_val<uint32_t>(15);
    const uint32_t qkv_k_tile         = get_arg_val<uint32_t>(16);
    const uint32_t qkv_v_tile         = get_arg_val<uint32_t>(17);
    const uint32_t z_head_offset      = get_arg_val<uint32_t>(18);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    constexpr uint32_t TILES_PER_COMPONENT = 4;  // 128 channels / 32

    const uint32_t tile_bytes = get_tile_size(cb_q);

    const auto state_acc     = TensorAccessor(accessor_args, state_addr, tile_bytes);
    const auto qkv_acc       = TensorAccessor(accessor_args, qkv_addr, tile_bytes);
    const auto z_acc         = TensorAccessor(accessor_args, z_addr, tile_bytes);
    const auto b_acc         = TensorAccessor(accessor_args, b_addr, tile_bytes);
    const auto a_acc         = TensorAccessor(accessor_args, a_addr, tile_bytes);
    const auto conv_state_ac = TensorAccessor(accessor_args, conv_state_addr, tile_bytes);
    const auto conv_w_acc    = TensorAccessor(accessor_args, conv_w_addr, tile_bytes);
    const auto a_log_acc     = TensorAccessor(accessor_args, a_log_addr, tile_bytes);
    const auto dt_acc        = TensorAccessor(accessor_args, dt_bias_addr, tile_bytes);
    const auto norm_acc      = TensorAccessor(accessor_args, norm_w_addr, tile_bytes);

    // ===================================================================
    // ONE-TIME READS (before token loop)
    // ===================================================================

    // 1. Read recurrent state [state_tiles tiles]
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

    // 2. Read conv_state and conv_weight into scratch (persistent)
    // Layout: [0..11] = conv_state (q+k+v), [12..23] = conv_weight (q+k+v)
    cb_reserve_back(cb_conv_scratch, 24);
    uint32_t scratch_base = get_write_ptr(cb_conv_scratch);
    uint32_t conv_w_base = scratch_base + 12 * tile_bytes;

    uint32_t comp_conv_tiles[3] = {conv_q_tile, conv_k_tile, conv_v_tile};

    // Read conv_state tiles [0..11]
    {
        uint32_t cs_l1 = scratch_base;
        for (uint32_t comp = 0; comp < 3; comp++) {
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_read_tile(comp_conv_tiles[comp] + t, conv_state_ac, cs_l1);
                cs_l1 += tile_bytes;
            }
        }
    }

    // Read conv_weight tiles [12..23]
    {
        uint32_t cw_l1 = conv_w_base;
        for (uint32_t comp = 0; comp < 3; comp++) {
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_read_tile(comp_conv_tiles[comp] + t, conv_w_acc, cw_l1);
                cw_l1 += tile_bytes;
            }
        }
    }
    noc_async_read_barrier();

    // 3-5: Pre-compute scaler and eps values for per-iteration push
    float inv_sqrt_dv = 1.0f / sqrtf(static_cast<float>(Dv));
    uint16_t sv_bf16 = f32_to_bf16(inv_sqrt_dv);

    // ===================================================================
    // PER-TOKEN LOOP
    // ===================================================================

    // Temporary buffer for reading qkv tiles of current token
    // We'll read into a small L1 area within the cb_conv_scratch
    // Actually we need separate space. Use stack-local approach:
    // Reserve space for 4 qkv tiles temporarily per component.
    // We can reuse the output CB areas since compute hasn't started yet for this token.

    uint32_t comp_cb[3] = {cb_q, cb_k, cb_v};
    uint32_t comp_qkv_base[3] = {qkv_q_tile, qkv_k_tile, qkv_v_tile};
    uint32_t k_write_addr = 0;

    uint32_t prev_qkv_tile_row = 0xFFFFFFFF;
    // For reading qkv tiles, we need temporary space. We'll read one component at a time.

    for (uint32_t s = 0; s < S; s++) {
        uint32_t tile_row = s / 32;
        uint32_t within_row = s % 32;

        // ---------------------------------------------------------------
        // Push fresh norm_w, scaler, eps each iteration (compute pops them)
        // ---------------------------------------------------------------
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
        {
            cb_reserve_back(cb_scaler, 1);
            uint32_t scaler_l1 = get_write_ptr(cb_scaler);
            volatile tt_l1_ptr uint16_t* stile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scaler_l1);
            for (uint32_t i = 0; i < 1024; i++) stile[i] = 0;
            for (uint32_t face = 0; face < 4; face++) {
                uint32_t base = face * 256;
                for (uint32_t c = 0; c < 16; c++) {
                    stile[base + c] = sv_bf16;
                }
            }
            cb_push_back(cb_scaler, 1);
        }
        {
            cb_reserve_back(cb_eps, 1);
            uint32_t eps_l1 = get_write_ptr(cb_eps);
            write_bf16_scalar_tile(eps_l1, 1e-6f);
            cb_push_back(cb_eps, 1);
        }

        // ---------------------------------------------------------------
        // A. Conv1d + SiLU for each component (q, k, v)
        // ---------------------------------------------------------------
        for (uint32_t comp = 0; comp < 3; comp++) {
            uint32_t cs_start = scratch_base + comp * TILES_PER_COMPONENT * tile_bytes;
            uint32_t cw_start = conv_w_base + comp * TILES_PER_COMPONENT * tile_bytes;

            // Read the current token's qkv projection for this component
            // From [1, 1, S, conv_dim] TILE: tile index = tile_row * qkv_col_tiles + comp_qkv_base[comp] + t
            cb_reserve_back(comp_cb[comp], TILES_PER_COMPONENT);
            uint32_t out_l1 = get_write_ptr(comp_cb[comp]);
            if (comp == 1) {
                k_write_addr = out_l1;
            }

            // Read qkv tiles for this component into the output area (temporarily).
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                uint32_t qkv_tile_idx = tile_row * qkv_col_tiles + comp_qkv_base[comp] + t;
                // Read this tile to a temporary spot (reuse the output tile area temporarily)
                uint32_t tmp_addr = out_l1 + t * tile_bytes;
                noc_async_read_tile(qkv_tile_idx, qkv_acc, tmp_addr);
            }
            noc_async_read_barrier();

            // Extract row within_row and get the 32 channel values per tile
            // Then do conv1d + silu
            volatile tt_l1_ptr uint16_t* cs_tile_ptrs[TILES_PER_COMPONENT];
            volatile tt_l1_ptr uint16_t* cw_tile_ptrs[TILES_PER_COMPONENT];
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                cs_tile_ptrs[t] = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs_start + t * tile_bytes);
                cw_tile_ptrs[t] = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cw_start + t * tile_bytes);
            }

            // First, extract each channel's new qkv value from the tile
            // Then update conv_state and compute conv1d output
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                uint32_t qkv_tile_addr = out_l1 + t * tile_bytes;
                uint32_t out_tile_addr = out_l1 + t * tile_bytes;

                for (uint32_t c = 0; c < 32; c++) {
                    // Extract new qkv value from row within_row, col c
                    float new_val = extract_bf16_element_2d(qkv_tile_addr, within_row, c);

                    // Read conv_state[c, 0..3]
                    uint32_t off0 = conv_tile_offset(c, 0);
                    float s0 = bf16_to_f32(cs_tile_ptrs[t][off0]);
                    float s1 = bf16_to_f32(cs_tile_ptrs[t][off0 + 1]);
                    float s2 = bf16_to_f32(cs_tile_ptrs[t][off0 + 2]);
                    float s3 = bf16_to_f32(cs_tile_ptrs[t][off0 + 3]);

                    // Read conv_weight[c, 0..3]
                    float w0 = bf16_to_f32(cw_tile_ptrs[t][off0]);
                    float w1 = bf16_to_f32(cw_tile_ptrs[t][off0 + 1]);
                    float w2 = bf16_to_f32(cw_tile_ptrs[t][off0 + 2]);
                    float w3 = bf16_to_f32(cw_tile_ptrs[t][off0 + 3]);

                    // Conv1d: roll(-1) + insert + dot product
                    float dot = s1 * w0 + s2 * w1 + s3 * w2 + new_val * w3;
                    float result = silu_f32(dot);

                    // Write result to output tile at row 0 col c (for compute kernel)
                    write_bf16_element_1d(out_tile_addr, c, f32_to_bf16(result));

                    // Update conv_state in-place (shift left, insert new_val)
                    cs_tile_ptrs[t][off0]     = f32_to_bf16(s1);
                    cs_tile_ptrs[t][off0 + 1] = f32_to_bf16(s2);
                    cs_tile_ptrs[t][off0 + 2] = f32_to_bf16(s3);
                    cs_tile_ptrs[t][off0 + 3] = f32_to_bf16(new_val);
                }
            }

            // L2-normalize q (comp=0) and k (comp=1)
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

            // Zero rows 1-31 in output tiles. The DRAM-read tile had data in
            // all rows (multi-token S-dimension), but conv1d only wrote row 0.
            // Stale rows cause matmul to produce non-zero results beyond row 0,
            // corrupting reduce_scalar in RMSNorm.
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                uint32_t addr = out_l1 + t * tile_bytes;
                volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
                float row0_vals[32];
                for (uint32_t c = 0; c < 32; c++) {
                    row0_vals[c] = extract_bf16_element_1d(addr, c);
                }
                for (uint32_t i = 0; i < 1024; i++) tile[i] = 0;
                for (uint32_t c = 0; c < 32; c++) {
                    write_bf16_element_1d(addr, c, f32_to_bf16(row0_vals[c]));
                }
            }

            cb_push_back(comp_cb[comp], TILES_PER_COMPONENT);
        }

        // ---------------------------------------------------------------
        // B. Construct k_T (transposed k) — use saved write addr, not get_read_ptr
        // ---------------------------------------------------------------
        {
            cb_reserve_back(cb_k_T, Dk_tiles);
            uint32_t k_src = k_write_addr;
            uint32_t k_T_dst = get_write_ptr(cb_k_T);
            for (uint32_t t = 0; t < Dk_tiles; t++) {
                volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_src);
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_T_dst);
                for (uint32_t i = 0; i < 1024; i++) dst[i] = 0;
                for (uint32_t j = 0; j < 16; j++) dst[j * 16] = src[j];
                for (uint32_t j = 0; j < 16; j++) dst[512 + j * 16] = src[256 + j];
                k_src += tile_bytes;
                k_T_dst += tile_bytes;
            }
            cb_push_back(cb_k_T, Dk_tiles);
        }

        // ---------------------------------------------------------------
        // C. Read b/a projections and compute decay + beta scalars
        // ---------------------------------------------------------------
        {
            // b_proj and a_proj are [1, 1, S, H_padded]. Token s at tile_row, within_row.
            uint32_t ba_tile_row_offset = tile_row * ba_col_tiles;
            uint32_t b_tile_idx = ba_tile_row_offset + head_idx / 32;
            uint32_t b_elem_col = head_idx % 32;

            cb_reserve_back(cb_beta, 1);
            uint32_t beta_l1 = get_write_ptr(cb_beta);
            noc_async_read_tile(b_tile_idx, b_acc, beta_l1);
            noc_async_read_barrier();
            float b_val = extract_bf16_element_2d(beta_l1, within_row, b_elem_col);

            cb_reserve_back(cb_g, 1);
            uint32_t g_l1 = get_write_ptr(cb_g);
            uint32_t a_tile_idx = ba_tile_row_offset + head_idx / 32;
            noc_async_read_tile(a_tile_idx, a_acc, g_l1);
            noc_async_read_barrier();
            float a_val = extract_bf16_element_2d(g_l1, within_row, head_idx % 32);

            // Read A_log and dt_bias (these are [1,1,1,H] — always tile_row 0)
            noc_async_read_tile(head_idx / 32, a_log_acc, beta_l1);
            noc_async_read_barrier();
            float a_log_val = extract_bf16_element_1d(beta_l1, head_idx % 32);

            noc_async_read_tile(head_idx / 32, dt_acc, beta_l1);
            noc_async_read_barrier();
            float dt_bias_val = extract_bf16_element_1d(beta_l1, head_idx % 32);

            float beta_val = 1.0f / (1.0f + expf(-b_val));
            float a_plus_dt = a_val + dt_bias_val;
            float sp = (a_plus_dt > 20.0f) ? a_plus_dt : logf(1.0f + expf(a_plus_dt));
            float decay_val = expf(-expf(a_log_val) * sp);

            write_bf16_scalar_tile(g_l1, decay_val);
            cb_push_back(cb_g, 1);

            write_bf16_scalar_tile(beta_l1, beta_val);
            cb_push_back(cb_beta, 1);
        }

        // ---------------------------------------------------------------
        // D. Read z tiles for token s
        // ---------------------------------------------------------------
        {
            // z_proj is [1, 1, S, H*Dv]. Token s: tile_row * z_col_tiles + z_head_offset + t
            uint32_t z_tile_row_offset = tile_row * z_col_tiles;
            cb_reserve_back(cb_z, Dv_tiles);
            uint32_t z_out_l1 = get_write_ptr(cb_z);

            for (uint32_t t = 0; t < Dv_tiles; t++) {
                uint32_t z_tile_idx = z_tile_row_offset + z_head_offset + t;
                uint32_t z_tile_addr = z_out_l1 + t * tile_bytes;
                noc_async_read_tile(z_tile_idx, z_acc, z_tile_addr);
            }
            noc_async_read_barrier();

            // Extract row within_row into row 0 and zero all other rows.
            // Non-row-0 data in z tiles causes incorrect silu(z) values that
            // interact with hardware tile operations.
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                uint32_t addr = z_out_l1 + t * tile_bytes;
                volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
                // Save row within_row values
                float row_vals[32];
                for (uint32_t c = 0; c < 32; c++) {
                    row_vals[c] = extract_bf16_element_2d(addr, within_row, c);
                }
                // Zero entire tile
                for (uint32_t i = 0; i < 1024; i++) tile[i] = 0;
                // Write saved values to row 0
                for (uint32_t c = 0; c < 32; c++) {
                    write_bf16_element_1d(addr, c, f32_to_bf16(row_vals[c]));
                }
            }

            cb_push_back(cb_z, Dv_tiles);
        }
    }

    // ===================================================================
    // FINAL: Push conv_state for writer
    // ===================================================================
    // The conv_state in scratch [0..11] has been updated across all S tokens.
    // Copy it to cb_conv_state_out and push once for the writer.
    {
        cb_reserve_back(cb_conv_state_out, 12);
        uint32_t conv_out_l1 = get_write_ptr(cb_conv_state_out);
        volatile tt_l1_ptr uint8_t* dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(conv_out_l1);
        volatile tt_l1_ptr uint8_t* src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch_base);
        for (uint32_t b = 0; b < 12 * tile_bytes; b++) {
            dst[b] = src[b];
        }
        cb_push_back(cb_conv_state_out, 12);
    }
}
