// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/complement_product.hpp"

// Every transform stays in complement form [E | B] (E = A - I). A block of `count` tiles in DST slots 0.. is packed to
// the FP32 stage buffer, to its exact copy when it has one (E), and to the send buffer; on the publishing group's last
// step also to the BF16 output row at `column`.
template <typename Stage, typename StageExact, typename Send, typename Pub>
FORCE_INLINE void pack_block(
    Stage& stage, StageExact* stage_exact, Send& send, Pub* published, uint32_t count, uint32_t tile, uint32_t column) {
    for (uint32_t slot = 0; slot < count; ++slot) {
        pack_tile(slot, stage.get_id(), tile + slot);
        if (stage_exact != nullptr) {
            pack_tile(slot, stage_exact->get_id(), tile + slot);
        }
        pack_tile(slot, send.get_id(), tile + slot);
    }
    if (published != nullptr) {
        pack_reconfig_data_format(published->get_id());
        for (uint32_t slot = 0; slot < count; ++slot) {
            pack_tile(slot, published->get_id(), column + slot);
        }
        pack_reconfig_data_format(stage.get_id());
    }
}

template <uint32_t Tiles, typename Stage, typename StageExact, typename Send>
FORCE_INLINE void reserve(Stage& stage, StageExact* stage_exact, Send& send) {
    stage.reserve_back(Tiles);
    if (stage_exact != nullptr) {
        stage_exact->reserve_back(Tiles);
    }
    send.reserve_back(Tiles);
}

template <uint32_t Tiles, typename Stage, typename StageExact, typename Send>
FORCE_INLINE void push(Stage& stage, StageExact* stage_exact, Send& send) {
    stage.push_back(Tiles);
    if (stage_exact != nullptr) {
        stage_exact->push_back(Tiles);
    }
    send.push_back(Tiles);
}

// Copy a group's [Rows, Columns] summary block unchanged. BF16 summaries pass a source register losslessly; FP32
// summaries unpack to DST. A single active group publishes its own summary, each tile row as it is staged.
template <
    uint32_t Rows,
    uint32_t Columns,
    typename In,
    typename Stage,
    typename StageExact,
    typename Send,
    typename Pub>
void copy_summary(In& in, Stage& stage, StageExact* stage_exact, Send& send, Pub* published) {
    constexpr uint32_t block = kda::largest_divisor_at_most(Columns, kda::dst_tile_capacity);
    const uint32_t in_id = in.get_id();
    reserve<Rows * Columns>(stage, stage_exact, send);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t row = 0; row < Rows; ++row) {
        if (published != nullptr) {
            published->reserve_back(Columns);
        }
        for (uint32_t column = 0; column < Columns; column += block) {
            const uint32_t tile = row * Columns + column;
            tile_regs_acquire();
            for (uint32_t slot = 0; slot < block; ++slot) {
                copy_tile(in_id, tile + slot, slot);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_block(stage, stage_exact, send, published, block, tile, column);
            tile_regs_release();
        }
        if (published != nullptr) {
            published->push_back(Columns);
        }
    }
    push<Rows * Columns>(stage, stage_exact, send);
}

// Compose a later transform (E2, B2) after an earlier one (E1, B1) in complement form, one [Kt, Columns] half at a
// time: E' = E2 + E1 + E2 @ E1 or B' = B2 + B1 + E2 @ B1 (complement_product.hpp).
template <
    uint32_t Kt,
    uint32_t Columns,
    typename Later,
    typename Earlier,
    typename LaterAddend,
    typename EarlierAddend,
    typename Stage,
    typename StageExact,
    typename Send,
    typename Pub>
void compose(
    Later& later,
    Earlier& earlier,
    LaterAddend& later_addend,
    EarlierAddend& earlier_addend,
    Stage& stage,
    StageExact* stage_exact,
    Send& send,
    Pub* published) {
    constexpr uint32_t block = kda::complement_block_columns<Columns>;
    reserve<Kt * Columns>(stage, stage_exact, send);
    reconfig_data_format(earlier.get_id(), later.get_id());
    for (uint32_t row = 0; row < Kt; ++row) {
        if (published != nullptr) {
            published->reserve_back(Columns);
        }
        for (uint32_t column = 0; column < Columns; column += block) {
            tile_regs_acquire();
            kda::complement_product_block<Kt, block>(
                kda::window<Kt>(later),
                kda::window<Columns>(earlier),
                kda::window<Columns>(later_addend),
                kda::window<Columns>(earlier_addend),
                row,
                column);
            tile_regs_commit();
            tile_regs_wait();
            pack_block(stage, stage_exact, send, published, block, row * Columns + column, column);
            tile_regs_release();
        }
        if (published != nullptr) {
            published->push_back(Columns);
        }
    }
    push<Kt * Columns>(stage, stage_exact, send);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_e, dfb::initial_b, dfb::stage_e);

    constexpr uint32_t e_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;
    DataflowBuffer initial_e(dfb::initial_e);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_e(dfb::stage_e);
    DataflowBuffer stage_e_exact(dfb::stage_e_exact);
    DataflowBuffer stage_b(dfb::stage_b);
    DataflowBuffer send_e(dfb::send_e);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_e(dfb::remote_e);
    DataflowBuffer remote_e_exact(dfb::remote_e_exact);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer remote_b_exact(dfb::remote_b_exact);
    DataflowBuffer output(dfb::output);
    // B has no exact stage copy: its stage buffer unpacks to DST and feeds only the addition.
    const auto no_exact_b = static_cast<decltype(&stage_b)>(nullptr);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t active = topology.head_groups(G);
    if (group >= active) {
        return;
    }
    auto* const published_copy = kda_chronology::publishes_unchanged(group, active) ? &output : nullptr;
    initial_e.wait_front(e_tiles);
    initial_b.wait_front(b_tiles);
    copy_summary<Kt, Kt>(initial_e, stage_e, &stage_e_exact, send_e, published_copy);
    copy_summary<Kt, Vt>(initial_b, stage_b, no_exact_b, send_b, published_copy);
    initial_e.pop_front(e_tiles);
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
        stage_e.wait_front(e_tiles);
        stage_e_exact.wait_front(e_tiles);
        stage_b.wait_front(b_tiles);
        remote_e.wait_front(e_tiles);
        remote_e_exact.wait_front(e_tiles);
        remote_b.wait_front(b_tiles);
        remote_b_exact.wait_front(b_tiles);
        auto* const published = kda_chronology::publishes_at_stage(group, active, distance) ? &output : nullptr;
        // This group's prefix (E2, B2) follows the remote one (E1, B1); B' reads E2 before the stage pops it.
        compose<Kt, Kt>(stage_e, remote_e, stage_e_exact, remote_e_exact, stage_e, &stage_e_exact, send_e, published);
        compose<Kt, Vt>(stage_e, remote_b, stage_b, remote_b_exact, stage_b, no_exact_b, send_b, published);
        stage_e.pop_front(e_tiles);
        stage_e_exact.pop_front(e_tiles);
        stage_b.pop_front(b_tiles);
        remote_e.pop_front(e_tiles);
        remote_e_exact.pop_front(e_tiles);
        remote_b.pop_front(b_tiles);
        remote_b_exact.pop_front(b_tiles);
    }
}
