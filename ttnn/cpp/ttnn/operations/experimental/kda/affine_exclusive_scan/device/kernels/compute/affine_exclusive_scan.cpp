// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

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
template <
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t AffineRowStride = Kt + Vt,
    typename AffineDFB,
    typename StateDFB,
    typename OutDFB>
FORCE_INLINE void matmul_add_affine_b(AffineDFB& affine, StateDFB& state, OutDFB& out) {
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
template <
    uint32_t Mt,
    uint32_t Kt,
    uint32_t At,
    uint32_t Vt,
    typename ADFB,
    typename AffineDFB,
    typename LocalBDFB,
    typename OutADFB,
    typename OutBDFB>
FORCE_INLINE void matmul_affine(ADFB& a, AffineDFB& affine, LocalBDFB& local_b, OutADFB& out_a, OutBDFB& out_b) {
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

// Inputs are waited for and popped by the caller; each helper reserves and publishes its whole output, and the packer
// keeps the format configured before the helper.
constexpr ckl::InputSpec resident(uint32_t dfb_id) {
    return ckl::input(dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
}

constexpr ckl::OutputSpec published(uint32_t dfb_id) {
    return ckl::output(dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
}

template <uint32_t In, uint32_t Out, typename InDFB, typename OutDFB>
FORCE_INLINE void copy(InDFB& in, OutDFB& out, uint32_t tiles) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(tiles).block_size(dst_tiles),
        ckl::CopyTile<resident(In)>{in},
        ckl::PackTile<published(Out)>{out});
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_a, dfb::initial_b, dfb::to_remote_a);

    constexpr uint32_t affine_a_tiles = Kt * Kt;
    constexpr uint32_t affine_b_tiles = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer local_a(dfb::local_a);
    DataflowBuffer local_b(dfb::local_b);
    DataflowBuffer to_remote_a(dfb::to_remote_a);
    DataflowBuffer to_remote_b(dfb::to_remote_b);
    DataflowBuffer from_remote_affine(dfb::from_remote_affine);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    DataflowBuffer tail_affine(dfb::tail_affine);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);
    DataflowBuffer reset_b(dfb::reset_b);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t active = topology.active_groups(G);
    if (group >= active) {
        return;
    }
    const uint32_t reset_group = topology.reset_group(G);
    initial_a.wait_front(affine_a_tiles);
    const bool reset_worker = group == reset_group;
    if (!reset_worker) {
        initial_b.wait_front(affine_b_tiles);
    }
    if (reset_worker) {
        const bool aligned_reset = topology.split_in_group(G) == 0;
        tail_entry_states.wait_front(affine_b_tiles);
        if (aligned_reset) {
            copy<dfb::tail_entry_states, dfb::reset_b>(tail_entry_states, reset_b, affine_b_tiles);
        } else {
            tail_affine.wait_front(affine_a_tiles + affine_b_tiles);
            matmul_add_affine_b<Kt, Kt, Vt>(tail_affine, tail_entry_states, reset_b);
        }
        reset_b.wait_front(affine_b_tiles);
        copy<dfb::initial_a, dfb::to_remote_a>(initial_a, to_remote_a, affine_a_tiles);
        copy<dfb::reset_b, dfb::to_remote_b>(reset_b, to_remote_b, affine_b_tiles);
        reset_b.pop_front(affine_b_tiles);
        if (!aligned_reset) {
            tail_affine.pop_front(affine_a_tiles + affine_b_tiles);
        }
        tail_entry_states.pop_front(affine_b_tiles);
    } else {
        copy<dfb::initial_a, dfb::to_remote_a>(initial_a, to_remote_a, affine_a_tiles);
        copy<dfb::initial_b, dfb::to_remote_b>(initial_b, to_remote_b, affine_b_tiles);
    }
    initial_a.pop_front(affine_a_tiles);
    if (!reset_worker) {
        initial_b.pop_front(affine_b_tiles);
    }

    for (uint32_t distance = 1; distance < active; distance *= 2) {
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

    initial_state.wait_front(affine_b_tiles);
    if (group == 0) {
        copy<dfb::initial_state, dfb::final>(initial_state, final, affine_b_tiles);
    } else {
        from_remote_affine.wait_front(affine_a_tiles + affine_b_tiles);
        matmul_add_affine_b<Kt, Kt, Vt>(from_remote_affine, initial_state, final);
        from_remote_affine.pop_front(affine_a_tiles + affine_b_tiles);
    }
    initial_state.pop_front(affine_b_tiles);
}
