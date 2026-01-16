// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Reader kernel for layernorm_fused_rm
// Phase 1: Generate scaler tile (1/W) for reduce operations
// Phase 2: Generate epsilon tile for numerical stability
// Phase 3: Read gamma (once, persistent)
// Phase 4: Read beta (once, persistent)
// Phase 5: Read input rows (per tile-row loop)

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // TensorAccessor compile-time args
    // Use next_compile_time_args_offset() to chain accessor args
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
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_5;

    // Tile dimensions
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t FACE_HEIGHT = 16;
    constexpr uint32_t FACE_WIDTH = 16;
    constexpr uint32_t TILE_SIZE_FACES = 4;  // 2x2 faces per tile

    // =====================================================================
    // Phase 1: Generate scaler tile (1/W) - 1 tile for reduce helper
    // The scaler is packed as float32, extract bfloat16 from upper 16 bits
    // =====================================================================
    {
        cb_reserve_back(cb_scaler, 1);
        uint32_t scaler_l1_addr = get_write_ptr(cb_scaler);

        // Convert float32 to bfloat16 (upper 16 bits)
        uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_packed >> 16);

        // Pack two bfloat16 values into one uint32 (little-endian: low | high)
        uint32_t scaler_packed_bf16 = (static_cast<uint32_t>(scaler_bf16) << 16) | scaler_bf16;

        // Fill the entire tile with the scaler value
        // Tile is organized as 4 faces (2x2), each face is 16x16 = 256 elements
        // Each element is bfloat16, so 256 elements = 512 bytes per face
        // We write 2 bfloat16 values per uint32, so 128 uint32 writes per face
        volatile tt_l1_ptr uint32_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_l1_addr);

        constexpr uint32_t elements_per_face = FACE_HEIGHT * FACE_WIDTH;      // 256
        constexpr uint32_t uint32_per_face = elements_per_face / 2;           // 128
        constexpr uint32_t total_uint32 = uint32_per_face * TILE_SIZE_FACES;  // 512

        for (uint32_t i = 0; i < total_uint32; ++i) {
            tile_ptr[i] = scaler_packed_bf16;
        }

        cb_push_back(cb_scaler, 1);
    }

    // =====================================================================
    // Phase 2: Generate epsilon tile - 1 tile
    // For add_tiles_bcast_cols, epsilon is broadcast from column 0
    // We fill the entire tile with epsilon for safety
    // =====================================================================
    {
        cb_reserve_back(cb_eps, 1);
        uint32_t eps_l1_addr = get_write_ptr(cb_eps);

        // Convert float32 epsilon to bfloat16
        uint16_t eps_bf16 = static_cast<uint16_t>(epsilon_packed >> 16);
        uint32_t eps_packed_bf16 = (static_cast<uint32_t>(eps_bf16) << 16) | eps_bf16;

        // Fill the entire tile with epsilon value
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
    // Phase 3: Read gamma (once) - 1 stick (1D tensor of width W)
    // =====================================================================
    {
        cb_reserve_back(cb_gamma_rm, 1);
        uint32_t gamma_l1_addr = get_write_ptr(cb_gamma_rm);
        noc_async_read(g.get_noc_addr(0), gamma_l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, 1);
    }

    // =====================================================================
    // Phase 4: Read beta (once) - 1 stick (1D tensor of width W)
    // =====================================================================
    {
        cb_reserve_back(cb_beta_rm, 1);
        uint32_t beta_l1_addr = get_write_ptr(cb_beta_rm);
        noc_async_read(b.get_noc_addr(0), beta_l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, 1);
    }

    // =====================================================================
    // Phase 5: Read input rows (per tile-row loop)
    // Each tile row is 32 sticks tall, each stick is W elements wide
    // CB page = one stick, push 32 pages per tile row
    // =====================================================================
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Read 32 sticks (one tile row height)
        cb_reserve_back(cb_in_rm, 32);
        uint32_t l1_addr = get_write_ptr(cb_in_rm);

        for (uint32_t stick_in_tile_row = 0; stick_in_tile_row < TILE_HEIGHT; stick_in_tile_row++) {
            noc_async_read(s.get_noc_addr(stick_id), l1_addr, stick_size);
            stick_id++;
            l1_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, 32);
    }
}
