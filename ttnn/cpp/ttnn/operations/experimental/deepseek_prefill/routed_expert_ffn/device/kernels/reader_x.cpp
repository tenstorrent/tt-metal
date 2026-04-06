// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BRISC dataflow kernel — reads x tiles into CB_IN0.
//
// Loop order matches the compute kernel: M_outer -> K_outer.
// For each (M_block, K_block) one batch of Mt_block_size × Kt_block_size
// tiles is pushed into CB_IN0.  The CB naturally back-pressures this kernel
// until the compute kernel pops the previous batch (which only happens after
// the N-inner loop has finished using those x tiles for both gate and up).
//
// Multi-core (2-D M×N split):
//   - m_blocks_local is a COMPILE-TIME constant — one kernel binary per M group.
//   - m_tile_start   is a RUNTIME arg — differs per core within the same M group.
//   - There is no N dependence: x is independent of N.  All n_n_cores cores in
//     the same M row read the same x tiles (necessary cost of N-splitting).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────────────
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t m_blocks_local = get_compile_time_arg_val(1);  // M blocks for this core group
    constexpr uint32_t Mt_block_size = get_compile_time_arg_val(2);
    constexpr uint32_t Kt_block_size = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(4);

    constexpr uint32_t ta_offset = 5;
    constexpr auto x_ta_args = TensorAccessorArgs<ta_offset>();

    // ── Runtime args ─────────────────────────────────────────────────────────
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);  // absolute tile offset for this core

    const auto x_acc = TensorAccessor(x_ta_args, x_addr, in0_tile_size);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t in0_block_size = Mt_block_size * Kt_block_size;
    constexpr uint32_t Kt_total = K_num_blocks * Kt_block_size;

    for (uint32_t m = 0; m < m_blocks_local; m++) {
        uint32_t m_tile_base = m_tile_start + m * Mt_block_size;

        for (uint32_t k = 0; k < K_num_blocks; k++) {
            uint32_t k_tile_base = k * Kt_block_size;

            cb_reserve_back(cb_in0, in0_block_size);
            uint32_t write_ptr = get_write_ptr(cb_in0);

            for (uint32_t mt = 0; mt < Mt_block_size; mt++) {
                for (uint32_t kt = 0; kt < Kt_block_size; kt++) {
                    uint32_t tile_id = (m_tile_base + mt) * Kt_total + (k_tile_base + kt);
                    noc_async_read_page(tile_id, x_acc, write_ptr);
                    write_ptr += in0_tile_size;
                }
            }
            noc_async_read_barrier();

            cb_push_back(cb_in0, in0_block_size);
        }
    }
}
