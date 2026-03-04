// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
//
// Reads row-major input sticks from DRAM and batches them as tile-sized RM pages
// into cb_in for the tilize compute helper. At program start, generates the
// reduce scaler (1/W) and epsilon scalar tiles.
//
// Compile-time args:
//   [0]  stick_size         -- bytes per input row (W * elem_size)
//   [1+] TensorAccessorArgs -- interleaved DRAM bank mapping for input
//
// Runtime args:
//   [0]  src_addr          -- input buffer DRAM base address
//   [1]  num_rows          -- N (total rows)
//   [2]  Wt               -- tiles per row (W / 32)
//   [3]  block_width_size  -- W * elem_size per tile-row block
//   [4]  has_weight        -- 1 if gamma provided
//   [5]  has_bias          -- 1 if beta provided
//   [6]  weight_addr       -- weight buffer address (0 if none)
//   [7]  bias_addr         -- bias buffer address (0 if none)
//   [8]  eps_bits          -- epsilon as bit-cast uint32

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Generate a scaler tile: fills row 0 of each face with the bfloat16 scaler value.
// The rest of the tile is zeroed using hardware zeros region.
inline void generate_bfloat16_scaler_tile(uint32_t cb_id, float scaler_val) {
    // Convert float to bfloat16 packed pair
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = scaler_val;
    uint16_t bf16 = static_cast<uint16_t>(conv.u >> 16);
    uint32_t packed = (static_cast<uint32_t>(bf16) << 16) | bf16;

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    // Zero the entire tile using hardware zeros
    // A bfloat16 tile is 4 faces * 16*16 * 2 bytes = 2048 bytes
    // MEM_ZEROS_SIZE is typically 256 bytes, so we need 2048/256 = 8 reads
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    constexpr uint32_t tile_bytes = 4 * 16 * 16 * 2;  // 2048 for bf16
    constexpr uint32_t num_zeros_reads = tile_bytes / MEM_ZEROS_SIZE;

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    uint32_t zero_addr = write_addr;
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, zero_addr);
        zero_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    // Fill row 0 of each face with the scaler value
    // Each face is 16x16 bf16 = 16*16*2 bytes = 512 bytes = 128 uint32_t
    // Row 0 is the first 16 bf16 = 8 uint32_t
    constexpr uint32_t face_size_u32 = 128;
    constexpr uint32_t row_size_u32 = 8;
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t face_offset = face * face_size_u32;
        for (uint32_t col = 0; col < row_size_u32; ++col) {
            ptr[face_offset + col] = packed;
        }
    }

    cb_push_back(cb_id, 1);
}

void kernel_main() {
    // ---- Runtime args ----
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t block_width_size = get_arg_val<uint32_t>(3);
    const uint32_t has_weight = get_arg_val<uint32_t>(4);
    const uint32_t has_bias = get_arg_val<uint32_t>(5);
    const uint32_t weight_addr = get_arg_val<uint32_t>(6);
    const uint32_t bias_addr = get_arg_val<uint32_t>(7);
    const uint32_t eps_bits = get_arg_val<uint32_t>(8);

    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);

    // TensorAccessor for the input tensor (args start at index 1)
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, stick_size);

    // CB indices
    constexpr uint32_t cb_in = 0;      // RM input pages
    constexpr uint32_t cb_eps = 1;     // epsilon scalar tile
    constexpr uint32_t cb_scaler = 2;  // reduce scaler (1/W)

    // ---- Generate reduce scaler (1/W) ----
    const uint32_t W = Wt * 32;
    const float scaler_val = 1.0f / static_cast<float>(W);
    generate_bfloat16_scaler_tile(cb_scaler, scaler_val);

    // ---- Generate epsilon tile ----
    union {
        uint32_t u;
        float f;
    } eps_union;
    eps_union.u = eps_bits;
    generate_bfloat16_scaler_tile(cb_eps, eps_union.f);

    // ---- Per tile-row: read 32 sticks and batch them as Wt tile-sized pages ----
    const uint32_t num_tile_rows = num_rows / 32;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Reserve space for Wt tile-sized pages in cb_in
        cb_reserve_back(cb_in, Wt);

        uint32_t l1_write_ptr = get_write_ptr(cb_in);

        // Pre-compute NoC addresses for the 32 sticks in this tile-row
        uint64_t base_src_noc_addr[32];
        for (uint32_t k = 0; k < 32; ++k) {
            uint32_t stick_id = tile_row * 32 + k;
            base_src_noc_addr[k] = get_noc_addr(stick_id, src);
        }

        // Issue async reads: for each of the 32 sticks, read block_width_size bytes
        for (uint32_t k = 0; k < 32; ++k) {
            noc_async_read(base_src_noc_addr[k], l1_write_ptr, block_width_size);
            l1_write_ptr += block_width_size;
        }

        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);
    }
}
