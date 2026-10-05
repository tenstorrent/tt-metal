// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Affine maps are carried in complement form, S -> S + E S + B with E = A - I (tt_metal_tracker-g1b.7): near-identity
// transitions of long-memory channels keep their relative precision in E, never in A.
//
// out = a @ b + x + y: the sum is preloaded into DST and the product accumulates onto it. FP32 half-DST holds four
// 32x32 output tiles. Production's four-tile output rows fit exactly and use one block per row; wider rows fall back
// to one tile per block because a row-major B operand cannot be column-sliced as a block without repacking.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt>
void matmul_onto_sum(
    DataflowBuffer& a,
    DataflowBuffer& b,
    DataflowBuffer& x,
    DataflowBuffer& y,
    DataflowBuffer& out,
    DataflowBuffer* send) {
    constexpr uint32_t max_block_columns = 4;
    constexpr uint32_t block_columns = Nt <= max_block_columns ? Nt : 1;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t x_id = x.get_id();
    const uint32_t y_id = y.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send == nullptr ? 0 : send->get_id();
    out.reserve_back(Mt * Nt);
    if (send != nullptr) {
        send->reserve_back(Mt * Nt);
    }
    for (uint32_t row = 0; row < Mt; row++) {
        for (uint32_t column = 0; column < Nt; column += block_columns) {
            tile_regs_acquire();
            reconfig_data_format(x_id, y_id);
            add_init(x_id, y_id);
            for (uint32_t offset = 0; offset < block_columns; offset++) {
                const uint32_t tile = row * Nt + column + offset;
                add_tiles(x_id, y_id, tile, tile, offset);
            }
            reconfig_data_format<SrcOrder::Reverse>(a_id, b_id);
            matmul_block_init(a_id, b_id, false, block_columns, 1, Kt);
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_block(a_id, b_id, row * Kt + k, k * Nt + column, 0, false, block_columns, 1, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t offset = 0; offset < block_columns; offset++) {
                const uint32_t out_tile = row * Nt + column + offset;
                pack_tile(offset, out_id, out_tile);
                if (send != nullptr) {
                    pack_tile(offset, send_id, out_tile);
                }
            }
            tile_regs_release();
        }
    }
    out.push_back(Mt * Nt);
    if (send != nullptr) {
        send->push_back(Mt * Nt);
    }
}

void copy(DataflowBuffer& in, DataflowBuffer& out, DataflowBuffer& send, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send.get_id();
    out.reserve_back(tiles);
    send.reserve_back(tiles);
    // Initial summaries may be BF16 while stage and send buffers use the canonical FP32 internal format. Copy updates
    // the source format; startup already configured the packer for the internal format.
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_tile(in_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id, tile);
        pack_tile(0, send_id, tile);
        tile_regs_release();
    }
    out.push_back(tiles);
    send.push_back(tiles);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_a, dfb::initial_b, dfb::stage_a);

    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_a(dfb::stage_a);
    DataflowBuffer stage_b(dfb::stage_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t active = topology.head_groups(G);
    if (group >= active) {
        return;
    }
    initial_a.wait_front(a_tiles);
    initial_b.wait_front(b_tiles);
    copy(initial_a, stage_a, send_a, a_tiles);
    copy(initial_b, stage_b, send_b, b_tiles);
    initial_a.pop_front(a_tiles);
    initial_b.pop_front(b_tiles);

    for (uint32_t distance = 1; distance < active; distance *= 2) {
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
        // Stage after remote: E = E_s + E_r + E_s E_r and B = B_s + B_r + E_s B_r. Both products read the old stage
        // E, which stays at the front until both new values are queued behind it.
        matmul_onto_sum<Kt, Kt, Kt>(stage_a, remote_a, stage_a, remote_a, stage_a, &send_a);
        matmul_onto_sum<Kt, Kt, Vt>(stage_a, remote_b, stage_b, remote_b, stage_b, &send_b);
        stage_a.pop_front(a_tiles);
        stage_b.pop_front(b_tiles);
        remote_a.pop_front(a_tiles);
        remote_b.pop_front(b_tiles);
    }
}
