// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
template <uint32_t tiles_per_row, uint32_t block_size>
TT_KERNEL void pack_untilize(uint32_t total_blocks) {
    constexpr uint32_t src_cb_id = dfb::src;
    constexpr uint32_t out_cb_id0 = dfb::untilize_out0;
    constexpr uint32_t out_cb_id1 = dfb::untilize_out1;

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

#ifndef ARCH_QUASAR
    // Gen1 packers honor the runtime output DFB, so preserve the single init/uninit around the loop.
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id0>();
#endif

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
#ifdef ARCH_QUASAR
                // The Quasar packer bakes the destination base at init, so reinitialize for each alternating output.
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
#else
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
#endif
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
#ifdef ARCH_QUASAR
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
#else
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
#endif
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

#ifndef ARCH_QUASAR
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
#endif
}
