// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the GGUF matmul for TQ2_0: the weight tiles are BFP2_b mantissas the matmul reads as -1, 0, +1
// (srca_forced_exp.h), with each K group's scales applied per window. Args: gguf_matmul_program_factory.cpp.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/srca_forced_exp.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "gguf_matmul_compute_common.hpp"

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t whole = 0, finish = 3;  // Role
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr bool one_row = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t window = get_compile_time_arg_val(3);
    constexpr uint32_t k_first = get_compile_time_arg_val(4);
    constexpr uint32_t cb_w = tt::CBIndex::c_0, cb_in0 = tt::CBIndex::c_1, cb_scale = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = gguf_matmul::cb_out, cb_own = gguf_matmul::cb_own;
    constexpr uint32_t record_pages = 33;               // c_0 pages of 64 B: eight 256-byte tiles, then the scales
    constexpr uint32_t in0_pages = one_row ? 64 : 1;   // c_1 pages per input tile
    constexpr uint32_t scale_dst = 2;
    const uint32_t segments = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);
    auto matmul_setup = []() {
        matmul_init(cb_in0, cb_w);
        srca_forced_exp_bfp2(127);
    };
    matmul_setup();

    // part *= the window's scales; one input row: column sums into row 0; then added to DEST tile 0 unless part is it.
    auto apply_scales = [&](uint32_t part) {
        cb_wait_front(cb_scale, 1);
        if constexpr (one_row) {
            srca_forced_exp_off();
            reconfig_data_format_srca(cb_w, cb_scale);
            copy_init(cb_scale);
            copy_tile(cb_scale, 0, scale_dst);
        } else {
            srca_forced_exp_off();
            reconfig_data_format_srca(cb_w, cb_scale);
            reconfig_data_format_srcb(cb_in0, cb_scale);
            unary_bcast_init<BroadcastType::ROW>(cb_scale);
            unary_bcast<BroadcastType::ROW>(cb_scale, 0, scale_dst);
        }
        mul_binary_tile_init();
        mul_binary_tile(part, scale_dst, part);
        if constexpr (one_row) {
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(part);
        }
        if (part != 0) {
            add_binary_tile_init();
            add_binary_tile(0, part, 0);
            fill_tile_init();
            fill_tile(part, 0.0f);
        }
        cb_pop_front(cb_scale, 1);
        reconfig_data_format_srca(cb_scale, cb_w);
        if constexpr (!one_row) {
            reconfig_data_format_srcb(cb_scale, cb_in0);
        }
        matmul_setup();
    };

    uint32_t in0_ready = 0;
    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t k0 = get_arg_val<uint32_t>(1 + 3 * s);
        const uint32_t k1 = get_arg_val<uint32_t>(2 + 3 * s);
        const uint32_t role = get_arg_val<uint32_t>(3 + 3 * s);
        if (role == finish) {
            // k0 is the count of the other cores' partials. Finishes come last.
            srca_forced_exp_off();
            reconfig_data_format_srca(cb_w, cb_own);
            gguf_matmul::finish_segment(k0);
            continue;
        }

        const uint32_t g0 = k0 / 8, g1 = k1 / 8;
        // A segment within one window accumulates in DEST tile 0, otherwise each window in tile 1, added to tile 0.
        const uint32_t part = g0 / window == (g1 - 1) / window ? 0 : 1;
        tile_regs_acquire();
        // DEST is not cleared between acquires: zero the accumulators.
        fill_tile_init();
        fill_tile(0, 0.0f);
        if (part != 0) {
            fill_tile(part, 0.0f);
        }
        if constexpr (one_row) {
            fill_tile(scale_dst, 0.0f);  // the [16, 32] scale tiles fill rows 0..15 of it only
        }
        matmul_setup();
        for (uint32_t g = g0; g < g1;) {
            const uint32_t end = (g / window + 1) * window < g1 ? (g / window + 1) * window : g1;
            for (; g < end; g++) {
                const uint32_t row = g % window;
                cb_wait_front(cb_w, record_pages);
                for (uint32_t i = 0; i < 8; i++) {
                    const uint32_t in0 = 8 * g + i - k_first;
                    if (in0 >= in0_ready) {
                        in0_ready = (in0 / chunk + 1) * chunk < k_tiles ? (in0 / chunk + 1) * chunk : k_tiles;
                        cb_wait_front(cb_in0, in0_ready * in0_pages);
                    }
                    matmul_tiles(cb_in0, cb_w, one_row ? in0_pages * in0 - row : in0, 4 * i, part);
                }
                cb_pop_front(cb_w, record_pages);
            }
            apply_scales(part);
        }
        gguf_matmul::pack_segment(role == whole);
    }
}
