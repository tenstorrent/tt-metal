// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
// Two-pass: reads each tile-row TWICE into cb_in (c_0).
// First copy: used for statistics (mean, variance)
// Second copy: used for normalization
// Generates reduce scaler tile (1/W) into cb_reduce_scaler (c_2).
// Generates epsilon tile into cb_eps (c_8).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma = tt::CBIndex::c_6;
constexpr uint32_t cb_beta = tt::CBIndex::c_7;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);
constexpr auto input_accessor_args = TensorAccessorArgs<3>();

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    uint32_t beta_addr = get_arg_val<uint32_t>(5);
    uint32_t eps_uint32 = get_arg_val<uint32_t>(6);

    // Input tensor accessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Compute W from stick_size (bfloat16 = 2 bytes per element)
    constexpr uint32_t element_size = 2;
    constexpr uint32_t W = stick_size / element_size;

    // Generate reduce scaler tile: 1/W for mean computation (AVG reduce)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce_scaler,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        W>();

    // Generate epsilon tile
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_uint32;
    float eps = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps);

    // Read gamma/beta if provided (persistent for entire program)
    // Gamma/beta are 1D [W] tensors stored as a single RM page of stick_size bytes.
    // We need to place them into tile format for ROW broadcast.
    //
    // TT tile layout (32x32 bf16): 4 faces of 16x16, stored as:
    //   Face 0 (top-left):     rows 0-15, cols 0-15  -> offset 0
    //   Face 1 (top-right):    rows 0-15, cols 16-31 -> offset 512
    //   Face 2 (bottom-left):  rows 16-31, cols 0-15 -> offset 1024
    //   Face 3 (bottom-right): rows 16-31, cols 16-31 -> offset 1536
    //
    // ROW broadcast reads Row0: first 16 elements from Face 0 row 0,
    // next 16 elements from Face 1 row 0.
    // So for each tile t: read gamma[t*32..t*32+15] to face0 offset, gamma[t*32+16..t*32+31] to face1 offset.
    constexpr uint32_t FACE_SIZE = 16;
    constexpr uint32_t face_row_bytes = FACE_SIZE * element_size;                  // 16 * 2 = 32 bytes
    constexpr uint32_t face_bytes = FACE_SIZE * FACE_SIZE * element_size;          // 16*16*2 = 512 bytes
    constexpr uint32_t tile_size_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;  // 32*32*2 = 2048

    if constexpr (has_gamma) {
        const InterleavedAddrGen<true> gamma_addrgen = {.bank_base_address = gamma_addr, .page_size = stick_size};
        uint64_t gamma_noc_base = gamma_addrgen.get_noc_addr(0);

        cb_reserve_back(cb_gamma, Wt);
        uint32_t gamma_l1 = get_write_ptr(cb_gamma);
        for (uint32_t t = 0; t < Wt; ++t) {
            uint32_t tile_base = gamma_l1 + t * tile_size_bytes;
            uint64_t src_base = gamma_noc_base + t * TILE_WIDTH * element_size;
            // First 16 elements -> Face 0, row 0
            noc_async_read(src_base, tile_base, face_row_bytes);
            // Next 16 elements -> Face 1, row 0
            noc_async_read(src_base + face_row_bytes, tile_base + face_bytes, face_row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    if constexpr (has_beta) {
        const InterleavedAddrGen<true> beta_addrgen = {.bank_base_address = beta_addr, .page_size = stick_size};
        uint64_t beta_noc_base = beta_addrgen.get_noc_addr(0);

        cb_reserve_back(cb_beta, Wt);
        uint32_t beta_l1 = get_write_ptr(cb_beta);
        for (uint32_t t = 0; t < Wt; ++t) {
            uint32_t tile_base = beta_l1 + t * tile_size_bytes;
            uint64_t src_base = beta_noc_base + t * TILE_WIDTH * element_size;
            // First 16 elements -> Face 0, row 0
            noc_async_read(src_base, tile_base, face_row_bytes);
            // Next 16 elements -> Face 1, row 0
            noc_async_read(src_base + face_row_bytes, tile_base + face_bytes, face_row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, Wt);
    }

    // Number of tile-rows
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;

    // Main loop: read 32 RM sticks per tile-row into cb_in TWICE
    // Pass 1 data: for statistics computation
    // Pass 2 data: for normalization
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        uint32_t row_start_stick = stick_id;

        // Push 1: for statistics pass
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(row_start_stick + s);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);

        // Push 2: for normalization pass
        cb_reserve_back(cb_in, Wt);
        l1_write_addr = get_write_ptr(cb_in);
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(row_start_stick + s);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);

        stick_id += TILE_HEIGHT;
    }
}
