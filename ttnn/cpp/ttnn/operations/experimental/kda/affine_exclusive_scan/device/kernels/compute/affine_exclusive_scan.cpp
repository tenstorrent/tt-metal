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

// out = first + second + later @ earlier over a [Kt, Columns] output, in complement-form blocks
// (complement_product.hpp).
template <
    uint32_t Kt,
    uint32_t Columns,
    typename Later,
    typename Earlier,
    typename First,
    typename Second,
    typename Out>
FORCE_INLINE void complement_product(
    const Later& later, const Earlier& earlier, const First& first, const Second& second, Out& out) {
    constexpr uint32_t block = kda::complement_block_columns<Columns>;
    const uint32_t out_id = out.get_id();
    out.reserve_back(Kt * Columns);
    reconfig_data_format(earlier.id(), later.id());
    for (uint32_t row = 0; row < Kt; ++row) {
        for (uint32_t column = 0; column < Columns; column += block) {
            tile_regs_acquire();
            kda::complement_product_block<Kt, block>(later, earlier, first, second, row, column);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t slot = 0; slot < block; ++slot) {
                pack_tile(slot, out_id, row * Columns + column + slot);
            }
            tile_regs_release();
        }
    }
    out.push_back(Kt * Columns);
}

template <typename InDFB, typename OutDFB>
FORCE_INLINE void copy(InDFB& in, OutDFB& out, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t first_tile = 0; first_tile < tiles; first_tile += kda::dst_tile_capacity) {
        const uint32_t batch_tiles =
            first_tile + kda::dst_tile_capacity <= tiles ? kda::dst_tile_capacity : tiles - first_tile;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < batch_tiles; ++tile) {
            copy_tile(in_id, first_tile + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < batch_tiles; ++tile) {
            pack_tile(tile, out_id, first_tile + tile);
        }
        tile_regs_release();
    }
    out.push_back(tiles);
}

// Every transform stays in complement form [E | B] (E = A - I); an entry state is S + E @ S + B. The state S and every
// B addend unpack to DST from exact FP32 copies. E is only ever a matmul operand here (the scan publishes states, not
// transforms), which reads it through a source register in any case, so E's own sums read it there too.
//
// The reset worker's transform discards everything before it: A = 0, so its E (-I) is never formed. Instead, a prefix
// that contains the reset is handled structurally, exactly as A = 0 would compose: composing after it keeps the later
// B and ignores the earlier one, and an entry state behind it is its B alone. Its E is never read.
template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_e, dfb::initial_b, dfb::to_remote_e);

    constexpr uint32_t e_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;
    constexpr uint32_t affine_row = Kt + Vt;
    DataflowBuffer initial_e(dfb::initial_e);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer local_e(dfb::local_e);
    DataflowBuffer local_b(dfb::local_b);
    DataflowBuffer to_remote_e(dfb::to_remote_e);
    DataflowBuffer to_remote_b(dfb::to_remote_b);
    DataflowBuffer from_remote_affine(dfb::from_remote_affine);
    DataflowBuffer from_remote_b_exact(dfb::from_remote_b_exact);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer initial_state_exact(dfb::initial_state_exact);
    DataflowBuffer final(dfb::final);
    DataflowBuffer tail_affine(dfb::tail_affine);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);
    DataflowBuffer tail_entry_states_exact(dfb::tail_entry_states_exact);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t active = topology.active_groups(G);
    if (group >= active) {
        return;
    }
    // G when there is no reset worker.
    const uint32_t reset_group = topology.reset_group(G);
    const bool reset_worker = group == reset_group;
    initial_e.wait_front(e_tiles);
    if (!reset_worker) {
        initial_b.wait_front(b_tiles);
    }
    if (reset_worker) {
        const bool aligned_reset = topology.split_in_group(G) == 0;
        tail_entry_states.wait_front(b_tiles);
        tail_entry_states_exact.wait_front(b_tiles);
        // The reset worker's own transform is (A = 0, seed); its E is never read, so the zeros dataflow wrote stand in.
        copy(initial_e, to_remote_e, e_tiles);
        if (aligned_reset) {
            copy(tail_entry_states_exact, to_remote_b, b_tiles);
        } else {
            // The seed is the tail transform applied to the tail entry state: S + E @ S + B.
            tail_affine.wait_front(Kt * affine_row);
            complement_product<Kt, Vt>(
                kda::window<affine_row>(tail_affine),
                kda::window<Vt>(tail_entry_states),
                kda::window<Vt>(tail_entry_states_exact),
                kda::window<affine_row, Kt>(tail_affine),
                to_remote_b);
            tail_affine.pop_front(Kt * affine_row);
        }
        tail_entry_states.pop_front(b_tiles);
        tail_entry_states_exact.pop_front(b_tiles);
    } else {
        copy(initial_e, to_remote_e, e_tiles);
        copy(initial_b, to_remote_b, b_tiles);
    }
    initial_e.pop_front(e_tiles);
    if (!reset_worker) {
        initial_b.pop_front(b_tiles);
    }

    for (uint32_t distance = 1; distance < active; distance *= 2) {
        if (group < distance) {
            continue;
        }
        local_e.wait_front(e_tiles);
        local_b.wait_front(b_tiles);
        from_remote_affine.wait_front(Kt * affine_row);
        from_remote_b_exact.wait_front(b_tiles);
        // This worker's prefix covers groups group - distance + 1 .. group and follows the remote one.
        if (reset_group <= group && group - reset_group < distance) {
            // It contains the reset, so it discards the remote prefix: (E2, B2) carries on unchanged.
            copy(local_e, to_remote_e, e_tiles);
            copy(local_b, to_remote_b, b_tiles);
        } else {
            // E' = E2 + E1 + E2 @ E1 and B' = B2 + B1 + E2 @ B1.
            complement_product<Kt, Kt>(
                kda::window<Kt>(local_e),
                kda::window<affine_row>(from_remote_affine),
                kda::window<Kt>(local_e),
                kda::window<affine_row>(from_remote_affine),
                to_remote_e);
            complement_product<Kt, Vt>(
                kda::window<Kt>(local_e),
                kda::window<affine_row, Kt>(from_remote_affine),
                kda::window<Vt>(local_b),
                kda::window<Vt>(from_remote_b_exact),
                to_remote_b);
        }
        local_e.pop_front(e_tiles);
        local_b.pop_front(b_tiles);
        from_remote_affine.pop_front(Kt * affine_row);
        from_remote_b_exact.pop_front(b_tiles);
    }

    initial_state.wait_front(b_tiles);
    initial_state_exact.wait_front(b_tiles);
    if (group == 0) {
        copy(initial_state_exact, final, b_tiles);
    } else {
        // The exclusive prefix covers groups 0 .. group - 1.
        from_remote_affine.wait_front(Kt * affine_row);
        from_remote_b_exact.wait_front(b_tiles);
        if (reset_group < group) {
            copy(from_remote_b_exact, final, b_tiles);
        } else {
            complement_product<Kt, Vt>(
                kda::window<affine_row>(from_remote_affine),
                kda::window<Vt>(initial_state),
                kda::window<Vt>(initial_state_exact),
                kda::window<Vt>(from_remote_b_exact),
                final);
        }
        from_remote_affine.pop_front(Kt * affine_row);
        from_remote_b_exact.pop_front(b_tiles);
    }
    initial_state.pop_front(b_tiles);
    initial_state_exact.pop_front(b_tiles);
}
