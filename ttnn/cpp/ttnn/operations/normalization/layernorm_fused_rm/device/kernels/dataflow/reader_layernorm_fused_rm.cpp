// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Reader kernel for layernorm_fused_rm
// Phase 1: Generate scaler tile (1/W) for reduce operations
// Phase 2: Generate epsilon tile for numerical stability
// Phase 3: Generate tiled gamma (replicated row) directly to c_6
// Phase 4: Generate tiled beta (replicated row) directly to c_7
// Phase 5: Read input rows (TWICE per tile-row for compute kernel needs)

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // TensorAccessor compile-time args
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto gamma_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(5);
    const uint32_t epsilon_packed = get_arg_val<uint32_t>(6);

    // Create TensorAccessors
    const auto s = TensorAccessor(src_args, src_addr, stick_size);
    const auto g = TensorAccessor(gamma_args, gamma_addr, stick_size);
    const auto b = TensorAccessor(beta_args, beta_addr, stick_size);

    // CB indices
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_4;     // Temporary: gamma RM data
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_5;      // Temporary: beta RM data
    constexpr uint32_t cb_gamma_tiled = tt::CBIndex::c_6;  // Tiled gamma (persistent)
    constexpr uint32_t cb_beta_tiled = tt::CBIndex::c_7;   // Tiled beta (persistent)

    // Tile dimensions
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t FACE_HEIGHT = 16;
    constexpr uint32_t FACE_WIDTH = 16;
    constexpr uint32_t TILE_SIZE_FACES = 4;  // 2x2 faces per tile

    // =====================================================================
    // Phase 1: Generate scaler tile (1/W) - 1 tile
    // =====================================================================
    {
        cb_reserve_back(cb_scaler, 1);
        uint32_t scaler_l1_addr = get_write_ptr(cb_scaler);

        // Convert float32 to bfloat16 (upper 16 bits)
        uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_packed >> 16);
        uint32_t scaler_packed_bf16 = (static_cast<uint32_t>(scaler_bf16) << 16) | scaler_bf16;

        volatile tt_l1_ptr uint32_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_l1_addr);

        constexpr uint32_t elements_per_face = FACE_HEIGHT * FACE_WIDTH;
        constexpr uint32_t uint32_per_face = elements_per_face / 2;
        constexpr uint32_t total_uint32 = uint32_per_face * TILE_SIZE_FACES;

        for (uint32_t i = 0; i < total_uint32; ++i) {
            tile_ptr[i] = scaler_packed_bf16;
        }

        cb_push_back(cb_scaler, 1);
    }

    // =====================================================================
    // Phase 2: Generate epsilon tile - 1 tile
    // =====================================================================
    {
        cb_reserve_back(cb_eps, 1);
        uint32_t eps_l1_addr = get_write_ptr(cb_eps);

        uint16_t eps_bf16 = static_cast<uint16_t>(epsilon_packed >> 16);
        uint32_t eps_packed_bf16 = (static_cast<uint32_t>(eps_bf16) << 16) | eps_bf16;

        volatile tt_l1_ptr uint32_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eps_l1_addr);

        constexpr uint32_t elements_per_face = FACE_HEIGHT * FACE_WIDTH;
        constexpr uint32_t uint32_per_face = elements_per_face / 2;
        constexpr uint32_t total_uint32 = uint32_per_face * TILE_SIZE_FACES;

        for (uint32_t i = 0; i < total_uint32; ++i) {
            tile_ptr[i] = eps_packed_bf16;
        }

        cb_push_back(cb_eps, 1);
    }

    // =====================================================================
    // Phase 3: Generate tiled gamma - Wt tiles
    // Read gamma row to temporary CB (c_4), then create tiles with row replicated
    // For ROW broadcast, we need tiles where each row has the same values
    // =====================================================================
    {
        // First read gamma row to temporary CB (c_4)
        cb_reserve_back(cb_gamma_rm, 1);
        uint32_t gamma_temp_l1 = get_write_ptr(cb_gamma_rm);

        // Read gamma stick from DRAM
        noc_async_read(g.get_noc_addr(0), gamma_temp_l1, stick_size);
        noc_async_read_barrier();

        // Now reserve space for tiled gamma in c_6
        cb_reserve_back(cb_gamma_tiled, Wt);
        uint32_t gamma_tiled_l1 = get_write_ptr(cb_gamma_tiled);

        // Source gamma data is at gamma_temp_l1 (bfloat16, W elements)
        volatile tt_l1_ptr uint16_t* gamma_src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gamma_temp_l1);

        // Tile size in bytes
        constexpr uint32_t tile_size_bytes = TILE_HEIGHT * TILE_WIDTH * 2;  // bfloat16

        // For each tile
        for (uint32_t t = 0; t < Wt; t++) {
            // Tile t starts at gamma[t * 32] and covers gamma[t*32 : (t+1)*32]
            uint32_t gamma_offset = t * TILE_WIDTH;
            uint32_t tile_l1_addr = gamma_tiled_l1 + t * tile_size_bytes;
            volatile tt_l1_ptr uint16_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);

            // For ROW broadcast pattern, fill each face
            // Tile layout: F0 (rows 0-15, cols 0-15), F1 (rows 0-15, cols 16-31),
            //              F2 (rows 16-31, cols 0-15), F3 (rows 16-31, cols 16-31)
            //
            // For ROW broadcast, each row should have the same gamma values:
            // row i, cols 0-15 -> gamma[t*32 + 0:16]
            // row i, cols 16-31 -> gamma[t*32 + 16:32]

            // Face 0 (top-left): rows 0-15, cols 0-15
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face0_idx = row * FACE_WIDTH + col;
                    tile_ptr[face0_idx] = gamma_src[gamma_offset + col];
                }
            }

            // Face 1 (top-right): rows 0-15, cols 16-31
            volatile tt_l1_ptr uint16_t* face1_ptr = tile_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face1_idx = row * FACE_WIDTH + col;
                    face1_ptr[face1_idx] = gamma_src[gamma_offset + FACE_WIDTH + col];
                }
            }

            // Face 2 (bottom-left): rows 16-31, cols 0-15
            volatile tt_l1_ptr uint16_t* face2_ptr = face1_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face2_idx = row * FACE_WIDTH + col;
                    face2_ptr[face2_idx] = gamma_src[gamma_offset + col];
                }
            }

            // Face 3 (bottom-right): rows 16-31, cols 16-31
            volatile tt_l1_ptr uint16_t* face3_ptr = face2_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face3_idx = row * FACE_WIDTH + col;
                    face3_ptr[face3_idx] = gamma_src[gamma_offset + FACE_WIDTH + col];
                }
            }
        }

        cb_push_back(cb_gamma_tiled, Wt);
        // We don't push/pop gamma_rm since it was only temporary - just release reservation
        // Actually, we need to pop it to free the space properly
        cb_push_back(cb_gamma_rm, 1);
        cb_pop_front(cb_gamma_rm, 1);
    }

    // =====================================================================
    // Phase 4: Generate tiled beta - Wt tiles
    // Same pattern as gamma
    // =====================================================================
    {
        // First read beta row to temporary CB (c_5)
        cb_reserve_back(cb_beta_rm, 1);
        uint32_t beta_temp_l1 = get_write_ptr(cb_beta_rm);

        // Read beta stick from DRAM
        noc_async_read(b.get_noc_addr(0), beta_temp_l1, stick_size);
        noc_async_read_barrier();

        // Now reserve space for tiled beta in c_7
        cb_reserve_back(cb_beta_tiled, Wt);
        uint32_t beta_tiled_l1 = get_write_ptr(cb_beta_tiled);

        volatile tt_l1_ptr uint16_t* beta_src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(beta_temp_l1);

        constexpr uint32_t tile_size_bytes = TILE_HEIGHT * TILE_WIDTH * 2;

        for (uint32_t t = 0; t < Wt; t++) {
            uint32_t beta_offset = t * TILE_WIDTH;
            uint32_t tile_l1_addr = beta_tiled_l1 + t * tile_size_bytes;
            volatile tt_l1_ptr uint16_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);

            // Face 0
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face0_idx = row * FACE_WIDTH + col;
                    tile_ptr[face0_idx] = beta_src[beta_offset + col];
                }
            }

            // Face 1
            volatile tt_l1_ptr uint16_t* face1_ptr = tile_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face1_idx = row * FACE_WIDTH + col;
                    face1_ptr[face1_idx] = beta_src[beta_offset + FACE_WIDTH + col];
                }
            }

            // Face 2
            volatile tt_l1_ptr uint16_t* face2_ptr = face1_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face2_idx = row * FACE_WIDTH + col;
                    face2_ptr[face2_idx] = beta_src[beta_offset + col];
                }
            }

            // Face 3
            volatile tt_l1_ptr uint16_t* face3_ptr = face2_ptr + FACE_HEIGHT * FACE_WIDTH;
            for (uint32_t row = 0; row < FACE_HEIGHT; row++) {
                for (uint32_t col = 0; col < FACE_WIDTH; col++) {
                    uint32_t face3_idx = row * FACE_WIDTH + col;
                    face3_ptr[face3_idx] = beta_src[beta_offset + FACE_WIDTH + col];
                }
            }
        }

        cb_push_back(cb_beta_tiled, Wt);
        cb_push_back(cb_beta_rm, 1);
        cb_pop_front(cb_beta_rm, 1);
    }

    // =====================================================================
    // Phase 5: Read input rows (TWICE per tile-row)
    // Compute kernel needs input data twice:
    // 1. First for mean/center/square/variance
    // 2. Second for re-center/normalize
    // =====================================================================
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Push input data TWICE per tile row
        for (uint32_t pass = 0; pass < 2; pass++) {
            cb_reserve_back(cb_in_rm, 32);
            uint32_t l1_addr = get_write_ptr(cb_in_rm);

            // Read 32 sticks (for pass 0 and pass 1 we read the same sticks)
            uint32_t read_stick_id = stick_id;  // Start from same position for both passes
            for (uint32_t stick_in_tile_row = 0; stick_in_tile_row < TILE_HEIGHT; stick_in_tile_row++) {
                noc_async_read(s.get_noc_addr(read_stick_id), l1_addr, stick_size);
                read_stick_id++;
                l1_addr += stick_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in_rm, 32);
        }

        // Advance stick_id after both passes
        stick_id += TILE_HEIGHT;
    }
}
