// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t largest_common_divisor_at_most(uint32_t lhs, uint32_t rhs, uint32_t limit) {
    for (uint32_t divisor = limit; divisor > 1; --divisor) {
        if (lhs % divisor == 0 && rhs % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

template <uint32_t Rows, uint32_t FirstColumns, uint32_t SecondColumns = FirstColumns>
struct MatmulSubblock {
    static constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // Columns cannot straddle packed output matrices, so they divide both widths. Rows consume the remaining DST
    // capacity. Selecting the largest legal divisor minimizes acquire/commit/wait cycles without partial blocks.
    static constexpr uint32_t columns = largest_common_divisor_at_most(FirstColumns, SecondColumns, dst_tiles);
    static constexpr uint32_t rows = largest_common_divisor_at_most(Rows, Rows, dst_tiles / columns);
    static_assert(rows * columns <= dst_tiles);
};

// Apply a packed affine pair to a state: out = affine_a * state + affine_b. Unlike matmul_affine, this emits only
// the transformed state; affine_b is loaded directly into DST so the matmul accumulates without an L1 partial.
template <uint32_t Mt, uint32_t Kt, uint32_t Vt, uint32_t AffineRowStride = Kt + Vt>
FORCE_INLINE void matmul_add_affine_b(DataflowBuffer& affine, DataflowBuffer& state, DataflowBuffer& out) {
    constexpr uint32_t subblock_cols = MatmulSubblock<Mt, Vt>::columns;
    constexpr uint32_t subblock_rows = MatmulSubblock<Mt, Vt>::rows;

    const uint32_t affine_id = affine.get_id();
    const uint32_t state_id = state.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(Mt * Vt);
    reconfig_data_format(state_id, affine_id);
    matmul_block_init(affine_id, state_id, false, subblock_cols, subblock_rows, Kt);
    for (uint32_t m = 0; m < Mt; m += subblock_rows) {
        for (uint32_t n = 0; n < Vt; n += subblock_cols) {
            tile_regs_acquire();
            reconfig_data_format_srca(state_id, affine_id);
            copy_init(affine_id);
            for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                    copy_tile(
                        affine_id,
                        (m + subblock_row) * AffineRowStride + Kt + n + subblock_col,
                        subblock_row * subblock_cols + subblock_col);
                }
            }
            reconfig_data_format_srca(affine_id, state_id);
            matmul_block_init(affine_id, state_id, false, subblock_cols, subblock_rows, Kt);
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_block(
                    affine_id,
                    state_id,
                    m * AffineRowStride + k,
                    k * Vt + n,
                    0,
                    false,
                    subblock_cols,
                    subblock_rows,
                    Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                    pack_tile(
                        subblock_row * subblock_cols + subblock_col,
                        out_id,
                        (m + subblock_row) * Vt + n + subblock_col);
                }
            }
            tile_regs_release();
        }
    }
    out.push_back(Mt * Vt);
}

// Compose packed affine pairs: out_a = a * affine_a and out_b = a * affine_b + local_b. The local B term is
// preloaded into DST before matmul accumulation, then the packed A and B columns are emitted to separate buffers.
template <uint32_t Mt, uint32_t Kt, uint32_t At, uint32_t Vt>
FORCE_INLINE void matmul_affine(
    DataflowBuffer& a, DataflowBuffer& affine, DataflowBuffer& local_b, DataflowBuffer& out_a, DataflowBuffer& out_b) {
    constexpr uint32_t Nt = At + Vt;
    constexpr uint32_t subblock_cols = MatmulSubblock<Mt, At, Vt>::columns;
    constexpr uint32_t subblock_rows = MatmulSubblock<Mt, At, Vt>::rows;

    const uint32_t a_id = a.get_id();
    const uint32_t affine_id = affine.get_id();
    const uint32_t local_b_id = local_b.get_id();
    const uint32_t out_a_id = out_a.get_id();
    const uint32_t out_b_id = out_b.get_id();
    out_a.reserve_back(Mt * At);
    out_b.reserve_back(Mt * Vt);
    reconfig_data_format(affine_id, a_id);
    matmul_block_init(a_id, affine_id, false, subblock_cols, subblock_rows, Kt);
    for (uint32_t m = 0; m < Mt; m += subblock_rows) {
        for (uint32_t n = 0; n < Nt; n += subblock_cols) {
            tile_regs_acquire();
            if (n >= At) {
                reconfig_data_format_srca(affine_id, local_b_id);
                copy_init(local_b_id);
                for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                    for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                        copy_tile(
                            local_b_id,
                            (m + subblock_row) * Vt + n - At + subblock_col,
                            subblock_row * subblock_cols + subblock_col);
                    }
                }
                reconfig_data_format_srca(local_b_id, affine_id);
                matmul_block_init(a_id, affine_id, false, subblock_cols, subblock_rows, Kt);
            }
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_block(a_id, affine_id, m * Kt + k, k * Nt + n, 0, false, subblock_cols, subblock_rows, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                    const uint32_t column = n + subblock_col;
                    const uint32_t dst = subblock_row * subblock_cols + subblock_col;
                    if (column < At) {
                        pack_tile(dst, out_a_id, (m + subblock_row) * At + column);
                    } else {
                        pack_tile(dst, out_b_id, (m + subblock_row) * Vt + column - At);
                    }
                }
            }
            tile_regs_release();
        }
    }
    out_a.push_back(Mt * At);
    out_b.push_back(Mt * Vt);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    constexpr uint32_t affine_a_tiles = Kt * Kt;
    constexpr uint32_t affine_b_tiles = Kt * Vt;
    DataflowBuffer local_a(dfb::local_a);
    DataflowBuffer local_b(dfb::local_b);
    DataflowBuffer to_remote_a(dfb::to_remote_a);
    DataflowBuffer to_remote_b(dfb::to_remote_b);
    DataflowBuffer from_remote_affine(dfb::from_remote_affine);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_a, dfb::initial_b, dfb::to_remote_a);
    ckl::copy<
        ckl::input(dfb::initial_a, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(dfb::to_remote_a, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(affine_a_tiles));
    ckl::copy<
        ckl::input(dfb::initial_b, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(dfb::to_remote_b, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(affine_b_tiles));

    for (uint32_t distance = 1; distance < G; distance *= 2) {
        if (group < distance) {
            continue;
        }
        local_a.wait_front(affine_a_tiles);
        local_b.wait_front(affine_b_tiles);
        from_remote_affine.wait_front(affine_a_tiles + affine_b_tiles);
        matmul_affine<Kt, Kt, Kt, Vt>(local_a, from_remote_affine, local_b, to_remote_a, to_remote_b);
        local_a.pop_front(affine_a_tiles);
        local_b.pop_front(affine_b_tiles);
        from_remote_affine.pop_front(affine_a_tiles + affine_b_tiles);
    }

    if (group == 0) {
        ckl::copy<
            ckl::input(
                dfb::initial_state, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::output(dfb::final, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(affine_b_tiles));
    } else {
        initial_state.wait_front(affine_b_tiles);
        from_remote_affine.wait_front(affine_a_tiles + affine_b_tiles);
        matmul_add_affine_b<Kt, Kt, Vt>(from_remote_affine, initial_state, final);
        from_remote_affine.pop_front(affine_a_tiles + affine_b_tiles);
        initial_state.pop_front(affine_b_tiles);
    }
}
