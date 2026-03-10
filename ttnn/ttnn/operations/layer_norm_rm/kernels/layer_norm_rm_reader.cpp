// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM sticks from DRAM into cb_rm_in (tilize input).
// Prepares reduce scaler (1/W) in cb_reduce_scaler.
// Prepares epsilon tile in cb_eps.
// Optionally reads gamma/beta tiles into cb_gamma/cb_beta (stage 3).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_rm_in = 0;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_gamma = 25;
constexpr uint32_t cb_beta = 26;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    // TensorAccessorArgs for input tensor follow at index 1+
    constexpr auto input_accessor_args = TensorAccessorArgs<1>();

    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t has_gamma = get_arg_val<uint32_t>(4);
    uint32_t has_beta = get_arg_val<uint32_t>(5);
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t eps_packed = get_arg_val<uint32_t>(8);

    // Input TensorAccessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ============ 1. Prepare reduce scaler: 1/W ============
    // W = Wt * 32 (tile width). Scaler = 1.0f / W for mean computation.
    float scaler_val = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(scaler_val);

    // ============ 2. Prepare epsilon tile ============
    // Fill cb_eps with epsilon value (packed bfloat16).
    // We use the same prepare_reduce_scaler pattern: zero the tile then fill row0.
    // eps_packed has bfloat16 in both halves of uint32.
    // We need to convert eps_packed back to float for prepare_reduce_scaler.
    // Actually, we can use prepare_reduce_scaler with the float epsilon value.
    // The eps_packed runtime arg is bfloat16 packed -- let's extract the float.
    // Since eps is typically 1e-5, we pass it through the same helper.
    // Extract bfloat16 from packed uint32 and convert to float:
    uint16_t eps_bf16 = static_cast<uint16_t>(eps_packed >> 16);
    uint32_t eps_f32_bits = static_cast<uint32_t>(eps_bf16) << 16;
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_f32_bits;
    float eps_float = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_float);

    // ============ 3. Read gamma/beta tiles (stage 3 only) ============
    // Gamma and beta are TILE_LAYOUT tensors on device, Wt tiles each.
    // For stages 1 and 2, has_gamma=0 and has_beta=0, so this is skipped.
    if (has_gamma) {
        // Gamma is RM layout (1,1,1,W) -- that's 1 stick of W elements.
        // But the compute kernel expects tiles. We need to read it as tiles.
        // Actually, gamma/beta are passed as RM tensors from host side.
        // The CB page size is tile_size (2048 bytes). We read Wt tiles.
        // For a (1,1,1,W) RM tensor, pages are sticks of size W*2 bytes.
        // We need to read the raw data into the CB.
        // Since gamma is a (1,1,1,W) RM tensor, it has 1 page of W*2 bytes.
        // But cb_gamma has Wt tile-sized pages. We need to tilize it.
        // Actually the compute kernel will handle gamma as-is in tile space.
        // For now, we read Wt pages of tile_size from the gamma buffer.
        // The gamma tensor must be in TILE_LAYOUT for this to work.
        // Let's handle this properly in stage 3.
        // PLACEHOLDER for stage 3
    }

    if (has_beta) {
        // PLACEHOLDER for stage 3
    }

    // ============ 4. Main loop: read RM sticks ============
    // Each block = 32 sticks = 1 tile-row.
    // We batch-read 32 sticks and push Wt tile-sized pages into cb_rm_in.
    uint32_t stick_id = start_stick_id;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Reserve Wt pages in cb_rm_in (32 sticks = Wt tiles worth of RM data)
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        // Read 32 RM sticks into the reserved CB space
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (compute sees Wt tile-sized pages)
        cb_push_back(cb_rm_in, Wt);
    }
}
