// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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

// Affine maps are carried in complement form, S -> S + E S + B with E = A - I. Long-memory channels have A within
// a few BF16/TF32 ulps of I; their E keeps its relative precision where A cannot (tt_metal_tracker-g1b.7).

// out = state + E state + B for a packed [E | B] pair. B is preloaded into DST, E state accumulates onto it, and the
// state itself is added last from its FP32 (UnpackToDest) view: the state is the long-lived carry, so it never passes
// through a source register.
template <uint32_t Mt, uint32_t Kt, uint32_t Vt, uint32_t AffineRowStride = Kt + Vt>
FORCE_INLINE void apply_complement_affine(
    DataflowBuffer& affine, DataflowBuffer& state, DataflowBuffer& state_exact, DataflowBuffer& out) {
    const uint32_t affine_id = affine.get_id();
    const uint32_t state_id = state.get_id();
    const uint32_t exact_id = state_exact.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(Mt * Vt);
    add_binary_tile_init();
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Vt; ++n) {
            tile_regs_acquire();
            reconfig_data_format_srca(affine_id);
            copy_init(affine_id);
            copy_tile(affine_id, m * AffineRowStride + Kt + n, 0);
            reconfig_data_format<SrcOrder::Reverse>(affine_id, state_id);
            matmul_block_init(affine_id, state_id, false, 1, 1, Kt);
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_block(affine_id, state_id, m * AffineRowStride + k, k * Vt + n, 0, false, 1, 1, Kt);
            }
            reconfig_data_format_srca(exact_id);
            copy_init(exact_id);
            copy_tile(exact_id, m * Vt + n, 1);
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_id, m * Vt + n);
            tile_regs_release();
        }
    }
    out.push_back(Mt * Vt);
}

// Compose packed complement pairs, local after remote:
//   out_e = local_e + remote_e + local_e remote_e,   out_b = local_b + remote_b + local_e remote_b.
// The sums are preloaded into DST and the product accumulates onto them; the packed E and B columns are emitted to
// separate buffers.
template <uint32_t Mt, uint32_t Kt, uint32_t At, uint32_t Vt>
FORCE_INLINE void compose_complement_affine(
    DataflowBuffer& local_e,
    DataflowBuffer& affine,
    DataflowBuffer& local_b,
    DataflowBuffer& out_e,
    DataflowBuffer& out_b) {
    constexpr uint32_t Nt = At + Vt;
    constexpr uint32_t subblock_cols = MatmulSubblock<Mt, At, Vt>::columns;
    constexpr uint32_t subblock_rows = MatmulSubblock<Mt, At, Vt>::rows;

    const uint32_t local_e_id = local_e.get_id();
    const uint32_t affine_id = affine.get_id();
    const uint32_t local_b_id = local_b.get_id();
    const uint32_t out_e_id = out_e.get_id();
    const uint32_t out_b_id = out_b.get_id();
    out_e.reserve_back(Mt * At);
    out_b.reserve_back(Mt * Vt);
    for (uint32_t m = 0; m < Mt; m += subblock_rows) {
        for (uint32_t n = 0; n < Nt; n += subblock_cols) {
            const bool e_columns = n < At;
            const uint32_t local_id = e_columns ? local_e_id : local_b_id;
            const uint32_t local_width = e_columns ? At : Vt;
            const uint32_t local_column = e_columns ? n : n - At;
            tile_regs_acquire();
            reconfig_data_format(local_id, affine_id);
            add_init(local_id, affine_id);
            for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                    add_tiles(
                        local_id,
                        affine_id,
                        (m + subblock_row) * local_width + local_column + subblock_col,
                        (m + subblock_row) * Nt + n + subblock_col,
                        subblock_row * subblock_cols + subblock_col);
                }
            }
            reconfig_data_format<SrcOrder::Reverse>(local_e_id, affine_id);
            matmul_block_init(local_e_id, affine_id, false, subblock_cols, subblock_rows, Kt);
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_block(local_e_id, affine_id, m * Kt + k, k * Nt + n, 0, false, subblock_cols, subblock_rows, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t subblock_row = 0; subblock_row < subblock_rows; ++subblock_row) {
                for (uint32_t subblock_col = 0; subblock_col < subblock_cols; ++subblock_col) {
                    const uint32_t column = n + subblock_col;
                    const uint32_t dst = subblock_row * subblock_cols + subblock_col;
                    if (column < At) {
                        pack_tile(dst, out_e_id, (m + subblock_row) * At + column);
                    } else {
                        pack_tile(dst, out_b_id, (m + subblock_row) * Vt + column - At);
                    }
                }
            }
            tile_regs_release();
        }
    }
    out_e.push_back(Mt * At);
    out_b.push_back(Mt * Vt);
}

FORCE_INLINE void copy(DataflowBuffer& in, DataflowBuffer& out, uint32_t tiles) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t first_tile = 0; first_tile < tiles; first_tile += dst_tiles) {
        const uint32_t batch_tiles = first_tile + dst_tiles <= tiles ? dst_tiles : tiles - first_tile;
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
    DataflowBuffer state(dfb::state);
    DataflowBuffer state_exact(dfb::state_exact);
    DataflowBuffer final(dfb::final);
    DataflowBuffer tail_affine(dfb::tail_affine);

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
        // The reset worker's transition starts from its tail seed: (E, B) = (-I, seed) for a group-aligned split,
        // else the tail summary applied to the seed. The state buffers hold the seed until it is consumed here.
        const bool aligned_reset = topology.split_in_group(G) == 0;
        state.wait_front(affine_b_tiles);
        state_exact.wait_front(affine_b_tiles);
        copy(initial_a, to_remote_a, affine_a_tiles);
        if (aligned_reset) {
            copy(state_exact, to_remote_b, affine_b_tiles);
        } else {
            tail_affine.wait_front(affine_a_tiles + affine_b_tiles);
            apply_complement_affine<Kt, Kt, Vt>(tail_affine, state, state_exact, to_remote_b);
            tail_affine.pop_front(affine_a_tiles + affine_b_tiles);
        }
        state.pop_front(affine_b_tiles);
        state_exact.pop_front(affine_b_tiles);
    } else {
        copy(initial_a, to_remote_a, affine_a_tiles);
        copy(initial_b, to_remote_b, affine_b_tiles);
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
        compose_complement_affine<Kt, Kt, Kt, Vt>(local_a, from_remote_affine, local_b, to_remote_a, to_remote_b);
        local_a.pop_front(affine_a_tiles);
        local_b.pop_front(affine_b_tiles);
        from_remote_affine.pop_front(affine_a_tiles + affine_b_tiles);
    }

    state.wait_front(affine_b_tiles);
    state_exact.wait_front(affine_b_tiles);
    if (group == 0) {
        copy(state_exact, final, affine_b_tiles);
    } else {
        from_remote_affine.wait_front(affine_a_tiles + affine_b_tiles);
        apply_complement_affine<Kt, Kt, Vt>(from_remote_affine, state, state_exact, final);
        from_remote_affine.pop_front(affine_a_tiles + affine_b_tiles);
    }
    state.pop_front(affine_b_tiles);
    state_exact.pop_front(affine_b_tiles);
}
