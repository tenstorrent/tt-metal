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

// The reader (taps [0, reader_taps), act_tile) and the writer (the other taps, act_tile_writer) supply, per channel
// tile, the four tap-shifted input tiles, and the reader, per channel block, the tap tiles (weights in row 0). Scaling
// a tile's columns by a tap is a matmul with that tap's diagonal, so the four taps accumulate in the destination
// registers with no partial round trip; SiLU then runs in place. The diagonals are built here once per channel block,
// as the identity times the row-broadcast tap.
template <uint32_t block_ct, uint32_t Mt, uint32_t group, uint32_t dest_limit>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    using qkv_conv::tap_count;
    static_assert(dest_limit <= compute_kernel_lib::DEST_AUTO_LIMIT);
    constexpr uint32_t weight_tiles = tap_count * block_ct;
    // Diagonals built per destination acquire.
    constexpr uint32_t diag_group = qkv_conv::dest_group(weight_tiles, dest_limit);
    using qkv_conv::reader_taps;
    using qkv_conv::writer_taps;
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer activation_writer(dfb::act_tile_writer);
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
            activation.wait_front(group * reader_taps);
            activation_writer.wait_front(group * writer_taps);
            output.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                // The taps accumulate in order 0, 1, 2, 3, whichever kernel made their shifted tile.
                for (uint32_t tap = 0; tap < reader_taps; ++tap) {
                    matmul_tiles(dfb::act_tile, dfb::weights, i * reader_taps + tap, tap * block_ct + ct0 + i, i);
                }
                for (uint32_t tap = reader_taps; tap < tap_count; ++tap) {
                    matmul_tiles(
                        dfb::act_tile_writer,
                        dfb::weights,
                        i * writer_taps + tap - reader_taps,
                        tap * block_ct + ct0 + i,
                        i);
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
            activation.pop_front(group * reader_taps);
            activation_writer.pop_front(group * writer_taps);
        }
    };
    // Walk the reader's order, which runs the tile row that waits for the predecessor's rows last.
    DataflowBuffer order(dfb::order);
    order.wait_front(1);
    const uint32_t deferred_row = read_tile_value(dfb::order, 0, 0);
    order.pop_front(1);
    qkv_conv::for_each_work_item(wi_start, wi_count, Mt, deferred_row, process);
    if (current_block != no_block) {
        weights.pop_front(weight_tiles);
    }
}
