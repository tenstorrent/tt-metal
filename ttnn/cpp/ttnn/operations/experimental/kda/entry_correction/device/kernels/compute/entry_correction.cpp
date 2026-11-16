// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

// output_c = F0_c + (FI_c - F0_c) @ seed per chunk, accumulated in the FP32 DST on top of the copied F0, and this
// core's tile rows of final = seed + (E @ seed + B) from the final transition's complement form E = A - I.
// The terms buffer holds a chunk row of F(0) at tile row, and of F(I) at fi_offset further: in place, the core's
// whole shard (core_chunks rows of F(0), then of F(I)); streamed, one chunk at a time, each formed into its
// difference just before its product.
// norm: each chunk output x is written as x / sqrt(mean(x^2) + epsilon) * weight * sigmoid(gate), x still in FP32.
template <uint32_t kt, uint32_t core_chunks, uint32_t streamed, uint32_t norm, uint32_t epsilon_bits>
TT_KERNEL void compute() {
    constexpr uint32_t state_tiles = kt * kt;
    constexpr uint32_t fi_offset = (streamed ? 1 : core_chunks) * kt;
    constexpr uint32_t terms_tiles = 2 * fi_offset;
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // With the norm, a chunk row is formed and normalized in blocks that leave as many DST slots free after them.
    constexpr uint32_t row_block = norm ? kda::largest_divisor_at_most(kt, dst_tiles / 2) : kt;
    DataflowBuffer seed_group(dfb::seed);
    DataflowBuffer seed_tail(dfb::tail);
    DataflowBuffer terms(dfb::terms);
    DataflowBuffer diff(dfb::diff);
    DataflowBuffer out(dfb::out);
    DataflowBuffer e(dfb::e);
    DataflowBuffer b(dfb::b);
    // The same seeds again, unpacked straight to DST (FP32, exact) for the final state.
    DataflowBuffer seed_group_exact(dfb::seed_exact);
    DataflowBuffer seed_tail_exact(dfb::tail_exact);
    DataflowBuffer final_rows(dfb::final_rows);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer weight_rows(dfb::weight_rows);
    DataflowBuffer x(dfb::x);
    DataflowBuffer square(dfb::square);
    DataflowBuffer inv_rms(dfb::inv_rms);
    DataflowBuffer scaler(dfb::scaler);
    const uint32_t terms_id = terms.get_id();
    const uint32_t diff_id = diff.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t e_id = e.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t final_id = final_rows.get_id();
    const uint32_t gate_id = gate.get_id();
    const uint32_t weight_rows_id = weight_rows.get_id();
    const uint32_t x_id = x.get_id();
    const uint32_t square_id = square.get_id();
    const uint32_t inv_rms_id = inv_rms.get_id();
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
    // The chunk output x and its square, normalized and gated into out.
    const auto normalize = [&]() {
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::square, dfb::scaler, dfb::inv_rms>(
                square,
                scaler,
                inv_rms,
                compute_kernel_lib::ReduceInputBlockShape::row(kt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                // The row means are in column 0, all the column broadcast reads.
                [](uint32_t dst) {
                    binop_with_scalar_tile_init();
                    add_unary_tile(dst, epsilon_bits, VectorMode::C);
                    rsqrt_tile_init();
                    rsqrt_tile<RsqrtMode::Fast>(dst, VectorMode::C);
                });
        // Per block, x * inv_rms on the FPU, sigmoid(gate) in the DST slots after it on the SFPU, their product, and
        // the weight on the FPU again from DST.
        x.wait_front(kt);
        inv_rms.wait_front(1);
        gate.wait_front(kt);
        out.reserve_back(kt);
        pack_reconfig_data_format(out_id);
        for (uint32_t first = 0; first < kt; first += row_block) {
            tile_regs_acquire();
            reconfig_data_format(x_id, inv_rms_id);
            mul_bcast_cols_init(x_id, inv_rms_id);
            for (uint32_t j = 0; j < row_block; ++j) {
                mul_tiles_bcast_cols(x_id, inv_rms_id, first + j, 0, j);
            }
            reconfig_data_format_srca(gate_id);
            copy_init(gate_id);
            for (uint32_t j = 0; j < row_block; ++j) {
                copy_tile(gate_id, first + j, row_block + j);
            }
            // The gate is BF16: its sigmoid needs only the BF16-accurate exp and one reciprocal step.
            sigmoid_tile_init();
            for (uint32_t j = 0; j < row_block; ++j) {
                sigmoid_tile<VectorMode::RC, false, false>(row_block + j);
            }
            mul_binary_tile_init();
            for (uint32_t j = 0; j < row_block; ++j) {
                mul_binary_tile(j, row_block + j, j);
            }
            reconfig_data_format(weight_rows_id, weight_rows_id);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_rows_id);
            for (uint32_t j = 0; j < row_block; ++j) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_rows_id, first + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < row_block; ++j) {
                pack_tile(j, out_id);
            }
            tile_regs_release();
        }
        out.push_back(kt);
        x.pop_front(kt);
        inv_rms.pop_front(1);
        gate.pop_front(kt);
    };
    // The differences first, for all of this core's chunks (they need no seed, so they run while the seeds arrive);
    // then per chunk F0's row is copied into DST and the K-sum accumulates on top of it.
    if (!streamed && chunks != 0) {
        terms.wait_front(terms_tiles);
        diff.reserve_back(chunks * kt);
        subtract(0, chunks * kt);
        diff.push_back(chunks * kt);
        diff.wait_front(chunks * kt);
    }
    if constexpr (norm) {
        // The weight (row 0 of its tiles) broadcast down every row, for the elementwise product with each output.
        weight.wait_front(kt);
        weight_rows.reserve_back(kt);
        pack_reconfig_data_format(weight_rows_id);
        reconfig_data_format(weight.get_id(), weight.get_id());
        unary_bcast_init<BroadcastType::ROW>(weight.get_id());
        for (uint32_t first = 0; first < kt; first += row_block) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < row_block; ++j) {
                unary_bcast<BroadcastType::ROW>(weight.get_id(), first + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < row_block; ++j) {
                pack_tile(j, weight_rows_id);
            }
            tile_regs_release();
        }
        weight_rows.push_back(kt);
        weight.pop_front(kt);
        weight_rows.wait_front(kt);
    }
    if (need_group) {
        seed_group.wait_front(state_tiles);
        seed_group_exact.wait_front(state_tiles);
    }
    if (need_tail) {
        seed_tail.wait_front(state_tiles);
        seed_tail_exact.wait_front(state_tiles);
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
        }
        const uint32_t result_id = norm ? x_id : out_id;
        if constexpr (norm) {
            x.reserve_back(kt);
            square.reserve_back(kt);
        } else {
            out.reserve_back(kt);
        }
        for (uint32_t first = 0; first < kt; first += row_block) {
            tile_regs_acquire();
            reconfig_data_format_srca(terms_id);
            copy_init(terms_id);
            for (uint32_t j = 0; j < row_block; ++j) {
                copy_tile(terms_id, row + first + j, j);
            }
            reconfig_data_format<SrcOrder::Reverse>(diff_id, seed);
            matmul_block_init(diff_id, seed, false, row_block, 1, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(diff_id, seed, row + k, k * kt + first, 0, false, row_block, 1, kt);
            }
            if constexpr (norm) {
                mul_binary_tile_init();
                for (uint32_t j = 0; j < row_block; ++j) {
                    mul_binary_tile(j, j, row_block + j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(result_id);
            for (uint32_t j = 0; j < row_block; ++j) {
                pack_tile(j, result_id);
            }
            if constexpr (norm) {
                pack_reconfig_data_format(square_id);
                for (uint32_t j = 0; j < row_block; ++j) {
                    pack_tile(row_block + j, square_id);
                }
            }
            tile_regs_release();
        }
        if constexpr (norm) {
            x.push_back(kt);
            square.push_back(kt);
            normalize();
        } else {
            out.push_back(kt);
        }
        if constexpr (streamed) {
            terms.pop_front(terms_tiles);
            diff.pop_front(kt);
        }
    }
    if (!streamed && chunks != 0) {
        terms.pop_front(terms_tiles);
        diff.pop_front(chunks * kt);
    }
    // This core's tile rows of final = seed + (E @ seed + B). B and the seed's row unpack to DST exactly and are added
    // on the SFPU, and E @ seed accumulates on top: only the small E @ seed term passes through the matmul's source
    // registers, so the long-memory seed is never truncated to their precision. Each pass forms `columns` tiles of the
    // row in DST slots [0, columns), with the seed's tiles in [columns, 2 columns).
    if (final) {
        constexpr uint32_t columns = kda::largest_divisor_at_most(kt, dst_tiles / 2);
        const uint32_t seed = final_tail ? tail_id : group_id;
        const uint32_t seed_exact = final_tail ? seed_tail_exact.get_id() : seed_group_exact.get_id();
        pack_reconfig_data_format(final_id);
        for (uint32_t row = first_row; row < kt; row += row_stride) {
            e.wait_front(kt);
            b.wait_front(kt);
            final_rows.reserve_back(kt);
            for (uint32_t first_column = 0; first_column < kt; first_column += columns) {
                tile_regs_acquire();
                reconfig_data_format_srca(b_id);
                copy_init(b_id);
                for (uint32_t c = 0; c < columns; ++c) {
                    copy_tile(b_id, first_column + c, c);
                }
                reconfig_data_format_srca(seed_exact);
                copy_init(seed_exact);
                for (uint32_t c = 0; c < columns; ++c) {
                    copy_tile(seed_exact, row * kt + first_column + c, columns + c);
                }
                add_binary_tile_init();
                for (uint32_t c = 0; c < columns; ++c) {
                    add_binary_tile(c, columns + c, c);
                }
                reconfig_data_format<SrcOrder::Reverse>(e_id, seed);
                matmul_block_init(e_id, seed, false, columns, 1, kt);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(e_id, seed, k, k * kt + first_column, 0, false, columns, 1, kt);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < columns; ++c) {
                    pack_tile(c, final_id);
                }
                tile_regs_release();
            }
            final_rows.push_back(kt);
            e.pop_front(kt);
            b.pop_front(kt);
        }
    }
    if (need_group) {
        seed_group.pop_front(state_tiles);
        seed_group_exact.pop_front(state_tiles);
    }
    if (need_tail) {
        seed_tail.pop_front(state_tiles);
        seed_tail_exact.pop_front(state_tiles);
    }
}
