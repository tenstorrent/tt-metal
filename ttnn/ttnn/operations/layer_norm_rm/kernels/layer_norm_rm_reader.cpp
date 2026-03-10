// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Reader Kernel
//
// Reads RM input sticks from DRAM and packs them into tile-sized CB pages
// for the tilize operation in the compute kernel.
//
// Compile-time args:
//   [0]  stick_size          — W * sizeof(bfloat16) bytes per RM stick
//   [1]  gamma_ct_start      — index where gamma TensorAccessorArgs begin (0 if no gamma)
//   [2]  beta_ct_start       — index where beta TensorAccessorArgs begin (0 if no beta)
//   [3+] TensorAccessorArgs(input) — interleaved input accessor
//   [?+] TensorAccessorArgs(gamma) — only if has_gamma
//   [?+] TensorAccessorArgs(beta)  — only if has_beta
//
// Runtime args:
//   [0] src_addr       — input buffer address
//   [1] start_stick_id — first RM stick for this core
//   [2] num_sticks     — nblocks * 32 sticks to read
//   [3] gamma_addr     — gamma buffer address (0 if absent)
//   [4] beta_addr      — beta buffer address (0 if absent)
//   [5] epsilon_packed — epsilon as uint32 IEEE-754 bits
//   [6] nblocks        — number of tile-row blocks

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Fill one full tile of bfloat16 with a scalar value.
FORCE_INLINE void fill_cb_with_val_bfloat16(uint32_t cb_id, uint32_t packed_scalar) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 512; ++i) {
        ptr[i] = packed_scalar;
    }
}

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t gamma_ct_start = get_compile_time_arg_val(1);
    constexpr uint32_t beta_ct_start = get_compile_time_arg_val(2);
    constexpr auto input_tensor_args = TensorAccessorArgs<3>();

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t epsilon_packed = get_arg_val<uint32_t>(5);
    const uint32_t nblocks = get_arg_val<uint32_t>(6);

    // ========== Constants ==========
    constexpr uint32_t cb_in_rm = 0;
    constexpr uint32_t cb_gamma = 6;
    constexpr uint32_t cb_beta = 7;
    constexpr uint32_t cb_eps = 8;
    constexpr uint32_t cb_scaler = 9;

    constexpr uint32_t Wt = stick_size / 64;
    constexpr bool has_gamma = gamma_ct_start != 0;
    constexpr bool has_beta = beta_ct_start != 0;

    // ========== Setup TensorAccessor ==========
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // ========== Fill reduce scaler CB (1/W) — program lifetime ==========
    constexpr uint32_t W = stick_size / 2;
    constexpr float scaler_val = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val);

    // ========== Fill epsilon CB — program lifetime ==========
    uint32_t eps_bf16 = epsilon_packed >> 16;
    uint32_t eps_packed_bf16 = (eps_bf16 << 16) | eps_bf16;
    cb_reserve_back(cb_eps, 1);
    fill_cb_with_val_bfloat16(cb_eps, eps_packed_bf16);
    cb_push_back(cb_eps, 1);

    // ========== Read gamma/beta once into L1 scratch area ==========
    // Gamma/beta are (1,1,1,W) RM tensors — 1 stick of W elements.
    // We read each stick once, then for each tile-row block, we
    // copy it 32 times into the CB to create proper tile-sized blocks.
    //
    // We use the CB itself as the scratch area: reserve Wt pages,
    // read the single stick into the first stick slot, then replicate.

    // For gamma/beta, we need to read one stick and replicate it 32 times
    // per tile-row block. Since the stick data is the same every time,
    // we read it once and then replicate from L1 for each block.

    // ========== Main loop: Read RM sticks ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < nblocks; ++block) {
        // Reserve Wt tile-sized pages in cb_in_rm
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks into the reserved space
        for (uint32_t s = 0; s < 32; ++s) {
            noc_async_read_page(stick_id, input_accessor, l1_write_addr);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, Wt);

        // Read gamma sticks for this block (same data every block)
        if constexpr (has_gamma) {
            constexpr auto gamma_tensor_args = TensorAccessorArgs<gamma_ct_start>();
            const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, stick_size);

            cb_reserve_back(cb_gamma, Wt);
            uint32_t gamma_l1_addr = get_write_ptr(cb_gamma);

            // Read stick 0 into first row
            noc_async_read_page(0, gamma_accessor, gamma_l1_addr);
            noc_async_read_barrier();

            // Replicate first stick to remaining 31 rows
            uint32_t src_addr_l1 = gamma_l1_addr;
            uint32_t dst_addr_l1 = gamma_l1_addr + stick_size;
            for (uint32_t s = 1; s < 32; ++s) {
                // Local L1-to-L1 copy via NOC loopback
                noc_async_read(get_noc_addr(src_addr_l1), dst_addr_l1, stick_size);
                dst_addr_l1 += stick_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, Wt);
        }

        // Read beta sticks for this block (same data every block)
        if constexpr (has_beta) {
            constexpr auto beta_tensor_args = TensorAccessorArgs<beta_ct_start>();
            const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, stick_size);

            cb_reserve_back(cb_beta, Wt);
            uint32_t beta_l1_addr = get_write_ptr(cb_beta);

            // Read stick 0 into first row
            noc_async_read_page(0, beta_accessor, beta_l1_addr);
            noc_async_read_barrier();

            // Replicate first stick to remaining 31 rows
            uint32_t src_addr_l1 = beta_l1_addr;
            uint32_t dst_addr_l1 = beta_l1_addr + stick_size;
            for (uint32_t s = 1; s < 32; ++s) {
                noc_async_read(get_noc_addr(src_addr_l1), dst_addr_l1, stick_size);
                dst_addr_l1 += stick_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_beta, Wt);
        }
    }
}
