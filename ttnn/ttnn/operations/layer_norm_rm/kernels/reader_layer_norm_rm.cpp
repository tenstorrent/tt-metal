// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads RM sticks from DRAM, fills epsilon/scaler CBs, optionally reads gamma/beta sticks

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

// Include fill_with_val_bfloat16 utility
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);          // tiles per row
    constexpr uint32_t Ht = get_compile_time_arg_val(2);          // total tile-rows
    constexpr uint32_t W = get_compile_time_arg_val(3);           // width in elements
    constexpr uint32_t has_gamma = get_compile_time_arg_val(4);   // 1 if gamma provided
    constexpr uint32_t has_beta = get_compile_time_arg_val(5);    // 1 if beta provided

    // TensorAccessor args for input start at index 6
    constexpr uint32_t input_ct_offset = 6;
    constexpr auto input_accessor_args = TensorAccessorArgs<input_ct_offset>();

    // Compute next offsets using chained TensorAccessorArgs
    // When has_gamma=0, gamma_ct_offset falls back to input_ct_offset (safe, won't read invalid args)
    // When has_gamma=1, gamma args actually exist at next_compile_time_args_offset()
    constexpr uint32_t gamma_ct_offset = has_gamma
                                             ? TensorAccessorArgs<input_ct_offset>::next_compile_time_args_offset()
                                             : input_ct_offset;  // Safe fallback: reuse input offset (never accessed)

    // CB indices
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_eps = 6;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_gamma = 9;
    constexpr uint32_t cb_beta = 10;

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t packed_eps = get_arg_val<uint32_t>(3);

    // ========== Setup TensorAccessor for input ==========
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ========== Fill epsilon CB (program lifetime) ==========
    cb_reserve_back(cb_eps, 1);
    fill_with_val_bfloat16(cb_eps, packed_eps);
    cb_push_back(cb_eps, 1);

    // ========== Generate reduce scaler (program lifetime) ==========
    // Use prepare_reduce_scaler with explicit 1/W value.
    // We use SUM reduce type in compute, so the scaler must incorporate the 1/W factor.
    // calculate_and_prepare_reduce_scaler with SUM ignores reduce_factor (sets scaler=1.0),
    // so we must use prepare_reduce_scaler directly.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / static_cast<float>(W));

    // ========== Read gamma if present (program lifetime) ==========
    if constexpr (has_gamma) {
        constexpr auto gamma_accessor_args = TensorAccessorArgs<gamma_ct_offset>();
        const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, stick_size);

        // Reserve Wt pages, zero the space, then write gamma stick into tile face row 0
        cb_reserve_back(cb_gamma, Wt);

        // Zero the entire CB space using MEM_ZEROS
        uint32_t gamma_write_addr = get_write_ptr(cb_gamma);
        {
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            uint32_t total_bytes = Wt * get_tile_size(cb_gamma);
            uint32_t addr = gamma_write_addr;
            uint32_t remaining = total_bytes;
            while (remaining > 0) {
                uint32_t chunk = (remaining > MEM_ZEROS_SIZE) ? MEM_ZEROS_SIZE : remaining;
                noc_async_read(zeros_noc_addr, addr, chunk);
                addr += chunk;
                remaining -= chunk;
            }
            noc_async_read_barrier();
        }

        // Write gamma values into face row 0 of each tile
        // Tile layout (bf16): face0 (16x16=512B), face1 (512B), face2 (512B), face3 (512B)
        // Row 0 of tile spans: face0[0..15] and face1[0..15] = 32 bf16 values
        constexpr uint32_t face_size_bytes = 16 * 16 * 2;  // 512 bytes per face for bf16
        constexpr uint32_t half_face_row = 16 * 2;         // 32 bytes = 16 bf16 elements

        for (uint32_t t = 0; t < Wt; t++) {
            uint32_t tile_start = gamma_write_addr + t * get_tile_size(cb_gamma);
            // Gamma is 1 stick (page 0). Tile t needs elements [t*32..t*32+31]
            uint64_t gamma_base = gamma_accessor.get_noc_addr(0);
            noc_async_read(gamma_base + t * 32 * 2, tile_start, half_face_row);
            noc_async_read(gamma_base + t * 32 * 2 + half_face_row, tile_start + face_size_bytes, half_face_row);
        }
        noc_async_read_barrier();

        cb_push_back(cb_gamma, Wt);
    }

    // ========== Compute beta offset safely ==========
    // Beta follows gamma in CT args. When has_beta=0, use input_ct_offset as safe fallback.
    constexpr uint32_t beta_ct_offset_val =
        has_beta ? (has_gamma ? TensorAccessorArgs<gamma_ct_offset>::next_compile_time_args_offset() : gamma_ct_offset)
                 : input_ct_offset;  // Safe fallback when has_beta=0

    // ========== Read beta if present (program lifetime) ==========
    if constexpr (has_beta) {
        constexpr auto beta_accessor_args = TensorAccessorArgs<beta_ct_offset_val>();
        const auto beta_accessor = TensorAccessor(beta_accessor_args, beta_addr, stick_size);

        cb_reserve_back(cb_beta, Wt);

        // Zero the entire CB space
        uint32_t beta_write_addr = get_write_ptr(cb_beta);
        {
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            uint32_t total_bytes = Wt * get_tile_size(cb_beta);
            uint32_t addr = beta_write_addr;
            uint32_t remaining = total_bytes;
            while (remaining > 0) {
                uint32_t chunk = (remaining > MEM_ZEROS_SIZE) ? MEM_ZEROS_SIZE : remaining;
                noc_async_read(zeros_noc_addr, addr, chunk);
                addr += chunk;
                remaining -= chunk;
            }
            noc_async_read_barrier();
        }

        constexpr uint32_t face_size_bytes = 16 * 16 * 2;
        constexpr uint32_t half_face_row = 16 * 2;

        for (uint32_t t = 0; t < Wt; t++) {
            uint32_t tile_start = beta_write_addr + t * get_tile_size(cb_beta);
            uint64_t beta_base = beta_accessor.get_noc_addr(0);
            noc_async_read(beta_base + t * 32 * 2, tile_start, half_face_row);
            noc_async_read(beta_base + t * 32 * 2 + half_face_row, tile_start + face_size_bytes, half_face_row);
        }
        noc_async_read_barrier();

        cb_push_back(cb_beta, Wt);
    }

    // ========== Main loop: read input RM sticks per tile-row ==========
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < Ht; ht++) {
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        // Read 32 contiguous RM sticks (= 1 tile-row)
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in, Wt);
    }
}
