// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"

// The reader supplies, per channel tile, the four tap-shifted input tiles and, per channel block, the tap tiles
// (weights in row 0). Scaling a tile's columns by a tap is a matmul with that tap's diagonal, so the four taps
// accumulate in the destination registers with no partial round trip; SiLU then runs in place. The diagonals are
// built here once per channel block, as the identity times the row-broadcast tap.
template <uint32_t block_ct, uint32_t Mt, uint32_t group, uint32_t dest_limit, uint32_t defer_first_rows>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    using qkv_conv::tap_count;
    static_assert(dest_limit <= compute_kernel_lib::DEST_AUTO_LIMIT);
    constexpr uint32_t weight_tiles = tap_count * block_ct;
    // Diagonals built per destination acquire.
    constexpr uint32_t diag_group = qkv_conv::dest_group(weight_tiles, dest_limit);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer output(dfb::output);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::act_tile, dfb::weights, dfb::output);
    // SiLU runs on the pack thread, on the destination half the math thread has just committed, so it overlaps the
    // next group's matmuls.
    ActivationInitHelper<KernelActivation::SILU>::init();
    eye.wait_front(1);

    constexpr uint32_t no_block = ~0U;
    uint32_t current_block = no_block;
    const auto process = [&](uint32_t work) {
        const uint32_t block = work / Mt;
        if (block != current_block) {
            if (current_block != no_block) {
                weights.pop_front(weight_tiles);
            }
            current_block = block;
            taps.wait_front(weight_tiles);
            weights.reserve_back(weight_tiles);
            mul_bcast_rows_init(dfb::eye, dfb::taps);
            for (uint32_t tile = 0; tile < weight_tiles; tile += diag_group) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < diag_group; ++i) {
                    mul_tiles_bcast_rows(dfb::eye, dfb::taps, 0, tile + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < diag_group; ++i) {
                    pack_tile<true>(i, dfb::weights, tile + i);
                }
                tile_regs_release();
            }
            weights.push_back(weight_tiles);
            taps.pop_front(weight_tiles);
            weights.wait_front(weight_tiles);
            matmul_init(dfb::act_tile, dfb::weights);
        }
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.wait_front(group * tap_count);
            output.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    matmul_tiles(dfb::act_tile, dfb::weights, i * tap_count + tap, tap * block_ct + ct0 + i, i);
                }
            }
            tile_regs_commit();
            // Waits for the math thread like tile_regs_wait.
            apply_activation_from_pack<KernelActivation::SILU>(group);
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, dfb::output);
            }
            tile_regs_release();
            output.push_back(group);
            activation.pop_front(group * tap_count);
        }
    };
    qkv_conv::for_each_work_item(wi_start, wi_count, Mt, defer_first_rows, process);
    if (current_block != no_block) {
        weights.pop_front(weight_tiles);
    }
}
