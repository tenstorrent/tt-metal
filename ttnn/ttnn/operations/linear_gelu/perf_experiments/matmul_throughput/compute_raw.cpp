// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_throughput bench: raw-LLK compute for linear_gelu (bypasses compute_kernel_lib::matmul_block).
// Same math as the op: per (row-block, n-block, subblock) one DEST window, K accumulated in fp32 DEST, GELU on the
// packer, packed subblock-major. Why raw: (1) the helper re-issues the per-call setup (shape decode, CB waits,
// per-subblock pack_reconfig_data_format) on each of the 45 calls/core; here the packer format is set once;
// (2) MT_BIAS_DEST seeds DEST with the row-broadcast bias tile (copy_tile) and runs K = 24 matmul steps instead of
// the bias-as-25th-K-tile fold, which the helper cannot express (its reload path is tied to interm spills).
// Compile switches: MT_NOGELU (ablation: NoneActivation), MT_BIAS_DEST (reader supplies a full row-broadcast
// bias tile at K position k_tiles of each chunk).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_w = get_compile_time_arg_val(4);
    constexpr uint32_t k_ext_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t gelu_fast = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles = k_ext_tiles - 1;

    const uint32_t core_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_n_blocks = get_arg_val<uint32_t>(1);
    const uint32_t sb_h = get_arg_val<uint32_t>(2);
    const uint32_t sb_w = get_arg_val<uint32_t>(3);
    const uint32_t tail_sb_h = get_arg_val<uint32_t>(4);
    const uint32_t tail_sb_w = get_arg_val<uint32_t>(5);

#ifdef MT_NOGELU
    using Gelu = NoneActivation;
#else
    using Gelu = GeluTanhActivation<gelu_fast>;
#endif
#ifdef MT_BIAS_DEST
    constexpr uint32_t k_steps = k_tiles;
#else
    constexpr uint32_t k_steps = k_ext_tiles;
#endif
    constexpr uint32_t act_block_tiles = block_h * k_ext_tiles;
    constexpr uint32_t w_chunk_tiles = k_ext_tiles * block_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    if constexpr (Gelu::activation != KernelActivation::NONE) {
        ActivationInitHelper<Gelu::activation, Gelu::param0>::init();
    }
    PACK((pack_reconfig_data_format(cb_out)));

    {
        MaybeDeviceZoneScope("compute_fill");
        cb_wait_front(cb_act, act_block_tiles);
        cb_wait_front(cb_w, w_chunk_tiles);
    }

    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;
    uint32_t init_sb_h = 0;
    uint32_t init_sb_w = 0;
    for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
        MaybeDeviceZoneScope("compute_matmul_gelu");
        const uint32_t first_row = rb * block_h;
        const bool tail = core_rows - first_row < block_h;
        const uint32_t rows = tail ? core_rows - first_row : block_h;
        const uint32_t cur_sb_h = tail ? tail_sb_h : sb_h;
        const uint32_t cur_sb_w = tail ? tail_sb_w : sb_w;
        const uint32_t sb_tiles = cur_sb_h * cur_sb_w;
        if (cur_sb_h != init_sb_h || cur_sb_w != init_sb_w) {
            init_sb_h = cur_sb_h;
            init_sb_w = cur_sb_w;
            ckernel::matmul_block_init(cb_act, cb_w, false, cur_sb_w, cur_sb_h, k_ext_tiles);
        }
        cb_wait_front(cb_act, act_block_tiles);
        for (uint32_t nb = 0; nb < num_n_blocks; ++nb) {
            // Resident W slice, never popped: cumulative wait from the fixed read pointer.
            cb_wait_front(cb_w, (nb + 1) * w_chunk_tiles);
            const uint32_t chunk_base = nb * w_chunk_tiles;
            for (uint32_t in0_sb = 0; in0_sb < rows / cur_sb_h; ++in0_sb) {
                for (uint32_t in1_sb = 0; in1_sb < block_w / cur_sb_w; ++in1_sb) {
                    tile_regs_acquire();
#ifdef MT_BIAS_DEST
                    copy_tile_to_dst_init_short(cb_w);
                    for (uint32_t r = 0; r < cur_sb_h; ++r) {
                        for (uint32_t c = 0; c < cur_sb_w; ++c) {
                            copy_tile(cb_w, chunk_base + k_tiles * block_w + in1_sb * cur_sb_w + c, r * cur_sb_w + c);
                        }
                    }
                    ckernel::matmul_block_init(cb_act, cb_w, false, cur_sb_w, cur_sb_h, k_ext_tiles);
#endif
                    uint32_t in0_idx = in0_sb * cur_sb_h * k_ext_tiles;
                    uint32_t in1_idx = chunk_base + in1_sb * cur_sb_w;
                    for (uint32_t k = 0; k < k_steps; ++k) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(
                            cb_act, cb_w, in0_idx, in1_idx, 0, false, cur_sb_w, cur_sb_h, k_ext_tiles);
#endif
                        ++in0_idx;
                        in1_idx += block_w;
                    }
                    tile_regs_commit();
                    cb_reserve_back(cb_out, sb_tiles);
                    if constexpr (Gelu::activation != KernelActivation::NONE) {
                        apply_activation_from_pack<Gelu::activation, Gelu::param0, Gelu::param1, Gelu::param2>(
                            sb_tiles);
                    } else {
                        tile_regs_wait();
                    }
                    for (uint32_t t = 0; t < sb_tiles; ++t) {
                        pack_tile(t, cb_out);
                    }
                    tile_regs_release();
                    cb_push_back(cb_out, sb_tiles);
                }
            }
        }
        cb_pop_front(cb_act, act_block_tiles);
    }
}
