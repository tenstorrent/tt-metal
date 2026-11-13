// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEEP IN SYNC WITH: pack_untilize.cpp (this directory)
//
// This is the Metal 2.0 fork of that kernel. Same logic, expressed against the Metal 2.0
// named-binding APIs: the three CB-index CTAs became DFB bindings, the two remaining CTAs and the
// block-count RTA became named arguments. A behavioural change to either one must be mirrored in the
// other.
//
// The fork exists because the legacy original is bound by factories that cannot all convert at once
// (sliding_window/halo on the legacy host API, experimental/padded_slice on Metal 2.0); it lives
// alongside the original rather than replacing it. Once the last legacy consumer is ported, delete the
// original and rename this file over it.
//
// Binding vocabulary a Metal 2.0 KernelSpec must supply for this source:
//   dfb::src            — tiled input DFB, bound CONSUMER
//   dfb::out0           — untilized output DFB for even-numbered blocks, bound PRODUCER
//   dfb::out1           — untilized output DFB for odd-numbered blocks, bound PRODUCER; only when the
//                         factory defines SEPARATE_OUT1. Without the define, odd blocks are packed into
//                         dfb::out0 as well (one DFB under both of the kernel's output names).
//   args::tiles_per_row — compile-time arg
//   args::block_size    — compile-time arg
//   args::total_blocks  — runtime arg

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t src_dfb = dfb::src;
    constexpr uint32_t out_dfb0 = dfb::out0;
#ifdef SEPARATE_OUT1
    constexpr uint32_t out_dfb1 = dfb::out1;
#else
    constexpr uint32_t out_dfb1 = out_dfb0;
#endif
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_arg(args::block_size);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg(args::total_blocks);

    compute_kernel_hw_startup(src_dfb, out_dfb0);

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<tiles_per_row, src_dfb, out_dfb0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // Use unified untilize with Neither mode since we handle init/uninit outside the loop
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_dfb,
                out_dfb0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_dfb,
                out_dfb1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_dfb, out_dfb0>();
}
