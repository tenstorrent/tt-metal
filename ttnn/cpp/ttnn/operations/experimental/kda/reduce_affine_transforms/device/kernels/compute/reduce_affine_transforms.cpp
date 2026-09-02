// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

// FP32 half-DST holds four 32x32 output tiles. Production's four-tile output rows fit exactly and use
// matmul_block; wider rows retain the tile loop because a row-major B operand cannot be column-sliced as a block
// without repacking.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt>
void matmul_product(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, DataflowBuffer* send) {
    constexpr uint32_t max_block_columns = 4;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send == nullptr ? 0 : send->get_id();
    out.reserve_back(Mt * Nt);
    if (send != nullptr) {
        send->reserve_back(Mt * Nt);
    }
    if constexpr (Nt <= max_block_columns) {
        matmul_block_init(a_id, b_id, false, Nt, 1, Kt);
        for (uint32_t row = 0; row < Mt; row++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_block(a_id, b_id, row * Kt + k, k * Nt, 0, false, Nt, 1, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t column = 0; column < Nt; column++) {
                const uint32_t out_tile = row * Nt + column;
                pack_tile(column, out_id, out_tile);
                if (send != nullptr) {
                    pack_tile(column, send_id, out_tile);
                }
            }
            tile_regs_release();
        }
    } else {
        matmul_init(a_id, b_id);
        for (uint32_t row = 0; row < Mt; row++) {
            for (uint32_t column = 0; column < Nt; column++) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < Kt; k++) {
                    matmul_tiles(a_id, b_id, row * Kt + k, k * Nt + column, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                const uint32_t out_tile = row * Nt + column;
                pack_tile(0, out_id, out_tile);
                if (send != nullptr) {
                    pack_tile(0, send_id, out_tile);
                }
                tile_regs_release();
            }
        }
    }
    out.push_back(Mt * Nt);
    if (send != nullptr) {
        send->push_back(Mt * Nt);
    }
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_a(dfb::stage_a);
    DataflowBuffer stage_b(dfb::stage_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer scratch(dfb::scratch);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_a, dfb::initial_b, dfb::stage_a);
    initial_a.wait_front(a_tiles);
    initial_b.wait_front(b_tiles);
    // initial summaries -> durable stage and send buffers
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(a_tiles),
        ckl::CopyTile<ckl::input(
            dfb::initial_a, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block)>{},
        ckl::PackTile<ckl::output(dfb::stage_a, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::PackTile<ckl::output(dfb::send_a, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(b_tiles),
        ckl::CopyTile<ckl::input(
            dfb::initial_b, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block)>{},
        ckl::PackTile<ckl::output(dfb::stage_b, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::PackTile<ckl::output(dfb::send_b, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
    initial_a.pop_front(a_tiles);
    initial_b.pop_front(b_tiles);

    for (uint32_t distance = 1; distance < G; distance *= 2) {
        // Every participating group produces a prefix consumed by a later group at a subsequent power-of-two
        // distance. Only the final group writes DRAM, but these intermediate prefixes are required inputs.
        if (group < distance) {
            continue;
        }
        // Stage buffers are durable state shared across independently progressing PACK, UNPACK, and NoC stages. The
        // current destination-register lifetime cannot span that synchronization boundary, so each prefix is queued
        // and reacquired here.
        stage_a.wait_front(a_tiles);
        stage_b.wait_front(b_tiles);
        remote_a.wait_front(a_tiles);
        remote_b.wait_front(b_tiles);
        reconfig_data_format(remote_a.get_id(), stage_a.get_id());
        // Both FP32 products remain separate calls: they consume different right-hand operands and publish
        // different output rectangles.
        matmul_product<Kt, Kt, Kt>(stage_a, remote_a, stage_a, &send_a);
        matmul_product<Kt, Kt, Vt>(stage_a, remote_b, scratch, nullptr);
        scratch.wait_front(b_tiles);
        // scratch + current affine offset -> updated stage and send buffers
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(b_tiles),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::scratch, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                ckl::input(dfb::stage_b, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block)>{},
            ckl::PackTile<ckl::output(dfb::stage_b, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
            ckl::PackTile<ckl::output(dfb::send_b, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
        stage_a.pop_front(a_tiles);
        stage_b.pop_front(b_tiles);
        remote_a.pop_front(a_tiles);
        remote_b.pop_front(b_tiles);
        scratch.pop_front(b_tiles);
    }
}
