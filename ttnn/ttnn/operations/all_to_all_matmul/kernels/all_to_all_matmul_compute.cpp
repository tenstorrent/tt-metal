// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul compute (TRISC0/1/2).
//
// Per head slot: for each of its n-blocks [nb0, nb1) (all of them, or one half under the tail split), matmul_block(A
// slot [block_m x Kt], weight block [Kt x block_n]) -> cb_out in TileRowMajor row groups of out_subblock_h x block_n.
// One K-block (R1): the K-sum stays in DEST, so cb_out is also the interm placeholder. A is retained across the head's
// n-blocks (WaitAndRetainOnLastBlock) and popped after the last one (release_slot); the weight block is waited and
// popped per call.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_a_landed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t block_m = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(4);
    constexpr uint32_t block_n = get_compile_time_arg_val(5);
    [[maybe_unused]] constexpr uint32_t num_n_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t slot_tiles = block_m * Kt;
    static_assert(block_m % out_subblock_h == 0 && block_n % out_subblock_w == 0, "subblocks must tile the block");

    const uint32_t n_slots = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a_landed, cb_weight, cb_out);
    CircularBuffer a_buf(cb_a_landed), w_buf(cb_weight), out_buf(cb_out);
    constexpr auto shape =
        MatmulBlockShape::of(block_m / out_subblock_h, block_n / out_subblock_w, out_subblock_h, out_subblock_w, Kt, 1);

    for (uint32_t s = 0; s < n_slots; ++s) {
        A2AMM_ZONE("cmp_head");
        const uint32_t nbw = get_arg_val<uint32_t>(1 + s);  // this core's n-blocks [nb0, nb1) of the slot's head
        for (uint32_t nb = nbw & 0xFFFF; nb < (nbw >> 16); ++nb) {
#ifdef ABLATE_NOCOMPUTE
            cb_wait_front(cb_a_landed, slot_tiles);
            cb_wait_front(cb_weight, Kt * block_n);
            cb_pop_front(cb_weight, Kt * block_n);
            for (uint32_t rg = 0; rg < block_m / out_subblock_h; ++rg) {
                cb_reserve_back(cb_out, out_subblock_h * block_n);
                cb_push_back(cb_out, out_subblock_h * block_n);
            }
            continue;
#endif
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::TileRowMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::WaitAndPopPerKBlock>(a_buf, w_buf, out_buf, out_buf, shape);
        }
        cb_pop_front(cb_a_landed, slot_tiles);  // release_slot
    }
}
