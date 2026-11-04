// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// output_c = F0_c + (FI_c - F0_c) @ seed per chunk, and this core's tile rows of final = A @ seed + B, accumulated in
// the FP32 DST on top of the copied F0 (or B).
// The terms buffer holds a chunk row of F(0) at tile row, and of F(I) at fi_offset further: in place, the core's
// whole shard (core_chunks rows of F(0), then of F(I)); streamed, one chunk at a time, each formed into its
// difference just before its product.
template <uint32_t kt, uint32_t core_chunks, uint32_t streamed>
TT_KERNEL void compute() {
    constexpr uint32_t state_tiles = kt * kt;
    constexpr uint32_t fi_offset = (streamed ? 1 : core_chunks) * kt;
    constexpr uint32_t terms_tiles = 2 * fi_offset;
    DataflowBuffer seed_group(dfb::seed);
    DataflowBuffer seed_tail(dfb::tail);
    DataflowBuffer terms(dfb::terms);
    DataflowBuffer diff(dfb::diff);
    DataflowBuffer out(dfb::out);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer final_rows(dfb::final_rows);
    const uint32_t terms_id = terms.get_id();
    const uint32_t diff_id = diff.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t final_id = final_rows.get_id();
    compute_kernel_hw_startup<SrcOrder::Reverse>(terms_id, seed_group.get_id(), out_id);

    uint32_t chunks;
    uint32_t group_chunks;
    uint32_t final;
    uint32_t final_tail;
    uint32_t need_group;
    uint32_t need_tail;
    uint32_t first_row;
    uint32_t row_stride;
    {
        DataflowBuffer ctl(dfb::ctl);
        ctl.wait_front(1);
        chunks = ctl.read_tile_value(0, 0);
        group_chunks = ctl.read_tile_value(0, 1);
        final = ctl.read_tile_value(0, 2);
        final_tail = ctl.read_tile_value(0, 3);
        need_group = ctl.read_tile_value(0, 4);
        need_tail = ctl.read_tile_value(0, 5);
        first_row = ctl.read_tile_value(0, 6);
        row_stride = ctl.read_tile_value(0, 7);
        ctl.pop_front(1);
    }
    // FI - F0 for tile rows [first, first + tiles) of the terms buffer, packed to the same rows of diff.
    const auto subtract = [&](uint32_t first, uint32_t tiles) {
        pack_reconfig_data_format(diff_id);
        reconfig_data_format(terms_id, terms_id);
        sub_init(terms_id, terms_id);
        for (uint32_t t = first; t < first + tiles; t += kt) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < kt; ++i) {
                sub_tiles(terms_id, terms_id, fi_offset + t + i, t + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < kt; ++i) {
                pack_tile(i, diff_id);
            }
            tile_regs_release();
        }
    };
    // The differences first, for all of this core's chunks (they need no seed, so they run while the seeds arrive);
    // then per chunk F0's row is copied into DST and the K-sum accumulates on top of it.
    if (!streamed && chunks != 0) {
        terms.wait_front(terms_tiles);
        diff.reserve_back(chunks * kt);
        subtract(0, chunks * kt);
        diff.push_back(chunks * kt);
        diff.wait_front(chunks * kt);
        pack_reconfig_data_format(out_id);
    }
    if (need_group) {
        seed_group.wait_front(state_tiles);
    }
    if (need_tail) {
        seed_tail.wait_front(state_tiles);
    }
    const uint32_t group_id = seed_group.get_id();
    const uint32_t tail_id = seed_tail.get_id();
    for (uint32_t c = 0; c < chunks; ++c) {
        const uint32_t seed = c < group_chunks ? group_id : tail_id;
        // Tile index of this chunk's row in the terms and difference buffers.
        const uint32_t row = streamed ? 0 : c * kt;
        if constexpr (streamed) {
            terms.wait_front(terms_tiles);
            diff.reserve_back(kt);
            subtract(0, kt);
            diff.push_back(kt);
            diff.wait_front(kt);
            pack_reconfig_data_format(out_id);
        }
        out.reserve_back(kt);
        tile_regs_acquire();
        reconfig_data_format_srca(terms_id);
        copy_init(terms_id);
        for (uint32_t j = 0; j < kt; ++j) {
            copy_tile(terms_id, row + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(diff_id, seed);
        matmul_block_init(diff_id, seed, false, kt, 1, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(diff_id, seed, row + k, k * kt, 0, false, kt, 1, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < kt; ++j) {
            pack_tile(j, out_id);
        }
        tile_regs_release();
        out.push_back(kt);
        if constexpr (streamed) {
            terms.pop_front(terms_tiles);
            diff.pop_front(kt);
        }
    }
    if (!streamed && chunks != 0) {
        terms.pop_front(terms_tiles);
        diff.pop_front(chunks * kt);
    }
    // This core's tile rows of final = A @ seed + B.
    if (final) {
        const uint32_t seed = final_tail ? tail_id : group_id;
        pack_reconfig_data_format(final_id);
        for (uint32_t row = first_row; row < kt; row += row_stride) {
            a.wait_front(kt);
            b.wait_front(kt);
            final_rows.reserve_back(kt);
            tile_regs_acquire();
            reconfig_data_format_srca(b_id);
            copy_init(b_id);
            for (uint32_t c = 0; c < kt; ++c) {
                copy_tile(b_id, c, c);
            }
            reconfig_data_format<SrcOrder::Reverse>(a_id, seed);
            matmul_block_init(a_id, seed, false, kt, 1, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(a_id, seed, k, k * kt, 0, false, kt, 1, kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < kt; ++r) {
                pack_tile(r, final_id);
            }
            tile_regs_release();
            final_rows.push_back(kt);
            a.pop_front(kt);
            b.pop_front(kt);
        }
    }
    if (need_group) {
        seed_group.pop_front(state_tiles);
    }
    if (need_tail) {
        seed_tail.pop_front(state_tiles);
    }
}
