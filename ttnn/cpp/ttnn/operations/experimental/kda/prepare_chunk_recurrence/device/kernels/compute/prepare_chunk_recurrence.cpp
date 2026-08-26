// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Chunk-parallel KDA prep. For cumulative vector gate G [C,K], factor
// exp(G_i-G_j) inside each q/k dot as exp(G_i)*exp(-G_j):
//   qd=q*exp(G), kl=k*exp(G), kr=k*exp(-G)
//   Akk=strictly_lower((beta*kl)@kr^T), Aqk=tril(qd@kr^T)
//   kd=beta*kl, k_dec_t=(kr*exp(G_last))^T, dl=exp(G_last).
// The existing blocked WY inverse helpers below are reused unchanged.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t largest_divisor_at_most(uint32_t value, uint32_t limit) {
    for (uint32_t divisor = limit; divisor > 1; --divisor) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

enum class ElementwiseBinaryOp { Add, Subtract, Multiply };

// Compute out[Mt,Nt] = A[Mt,Kt] @ (transpose_b ? B[Nt,Kt]^T : B[Kt,Nt]) in the largest rectangular
// subblocks that exactly divide the output and fit in destination registers.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr>
inline void matmul_blocks(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o) {
    constexpr uint32_t subblock_columns = largest_divisor_at_most(Nt, max_dst_tiles);
    constexpr uint32_t subblock_rows = largest_divisor_at_most(Mt, max_dst_tiles / subblock_columns);
    static_assert(subblock_rows * subblock_columns <= max_dst_tiles);
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
    pack_reconfig_data_format(o_id);  // mixed bf16/fp32 CBs: set packer to this output's format
    // Matmul maps in0=a_id->srcB and in1=b_id->srcA. Reconfigure unpack formats explicitly because init only
    // asserts formats; otherwise mixed fp32/bf16 CBs are read in the wrong format.
    reconfig_data_format(b_id, a_id);
    matmul_block_init(a_id, b_id, Tr, subblock_columns, subblock_rows, Kt);
    for (uint32_t mi = 0; mi < Mt; mi += subblock_rows) {
        for (uint32_t ni = 0; ni < Nt; ni += subblock_columns) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                // kt_dim describes operand geometry; each K slice is still issued explicitly. B^T is stored
                // [Nt,Kt], while the non-transposed B is stored [Kt,Nt].
                const uint32_t b_index = Tr ? ni * Kt + ki : ki * Nt + ni;
                matmul_block(a_id, b_id, mi * Kt + ki, b_index, 0, Tr, subblock_columns, subblock_rows, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < subblock_rows; row++) {
                for (uint32_t column = 0; column < subblock_columns; column++) {
                    pack_tile(row * subblock_columns + column, o_id, (mi + row) * Nt + ni + column);
                }
            }
            tile_regs_release();
        }
    }
    o.push_back(Mt * Nt);
}

// Apply a typed binary operation tilewise, batching each destination-register synchronization.
template <ElementwiseBinaryOp Op>
inline void elementwise_binary(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o, uint32_t n) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format(a_id, b_id);  // binary(a_id,b_id): a_id->srcA, b_id->srcB
    if constexpr (Op == ElementwiseBinaryOp::Add) {
        add_init(a_id, b_id);
    } else if constexpr (Op == ElementwiseBinaryOp::Subtract) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t input_tile = block_start + tile;
            if constexpr (Op == ElementwiseBinaryOp::Add) {
                add_tiles(a_id, b_id, input_tile, input_tile, tile);
            } else if constexpr (Op == ElementwiseBinaryOp::Subtract) {
                sub_tiles(a_id, b_id, input_tile, input_tile, tile);
            } else {
                mul_tiles(a_id, b_id, input_tile, input_tile, tile);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

inline void square_tiles(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    square_tile_init();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            square_tile(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

inline void exponential_tiles(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);  // unary: in_id->srcA
    copy_tile_to_dst_init_short(in_id);
    exp_tile_init();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            exp_tile(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

inline void multiply_by_half(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    constexpr uint32_t fp32_half_bits = __builtin_bit_cast(uint32_t, 0.5F);
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    binop_with_scalar_tile_init();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            mul_unary_tile(tile, fp32_half_bits);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

inline void negated_exponential_tiles(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
        }
        negative_tile_init();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            negative_tile(tile);
        }
        exp_tile_init();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            exp_tile(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void multiply_by_column(DataflowBuffer& a, DataflowBuffer& col, DataflowBuffer& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t col_id = col.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
    pack_reconfig_data_format(o_id);
    reconfig_data_format(a_id, col_id);  // bcast(a_id,col_id): a_id->srcA, col_id->srcB
    mul_bcast_cols_init(a_id, col_id);
    const uint32_t output_tiles = Mt * Nt;
    for (uint32_t block_start = 0; block_start < output_tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles =
            (output_tiles - block_start < max_dst_tiles) ? output_tiles - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t input_tile = block_start + tile;
            mul_tiles_bcast_cols(a_id, col_id, input_tile, input_tile / Nt, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(Mt * Nt);
}

// Copy one source tile into a one-tile output buffer.
inline void copy_tile_to_buffer(DataflowBuffer& src, uint32_t src_tile, DataflowBuffer& o) {
    const uint32_t src_id = src.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(1);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(src_id);
    copy_tile_to_dst_init_short(src_id);
    tile_regs_acquire();
    copy_tile(src_id, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o_id, 0);
    tile_regs_release();
    o.push_back(1);
}

// Invert (I-N) for a strictly-lower 32x32 N using the exact nilpotent product
// (I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16). Eight tile matmuls replace
// the masked 16x16 Horner path's thirty full-tile matmuls; the shorter dependency chain is also
// expected to improve fp32 stability, but that must be validated empirically.
inline void invert_doubling(
    DataflowBuffer& src,
    uint32_t tile,
    DataflowBuffer& out,
    DataflowBuffer& state,
    DataflowBuffer& final_state,
    DataflowBuffer& state_two,
    DataflowBuffer& state_three,
    DataflowBuffer& eye) {
    DataflowBuffer* power = &state;
    DataflowBuffer* sum = &final_state;
    DataflowBuffer* next_power = &state_two;
    DataflowBuffer* product = &state_three;

    copy_tile_to_buffer(src, tile, *power);
    power->wait_front(1);
    elementwise_binary<ElementwiseBinaryOp::Add>(eye, *power, *sum, 1);
    sum->wait_front(1);
    matmul_blocks<1, 1, 1, false>(*power, *power, *next_power);
    next_power->wait_front(1);
    power->pop_front(1);
    power = next_power;
    next_power = &state;

    for (uint32_t step = 0; step < 4; ++step) {
        matmul_blocks<1, 1, 1, false>(*power, *sum, *product);
        product->wait_front(1);
        if (step < 3) {
            matmul_blocks<1, 1, 1, false>(*power, *power, *next_power);
            next_power->wait_front(1);
        }
        power->pop_front(1);
        elementwise_binary<ElementwiseBinaryOp::Add>(*sum, *product, *power, 1);
        power->wait_front(1);
        sum->pop_front(1);
        product->pop_front(1);
        if (step < 3) {
            DataflowBuffer* old_sum = sum;
            sum = power;
            power = next_power;
            next_power = old_sum;
        } else {
            sum = power;
        }
    }
    copy_tile_to_buffer(*sum, 0, out);
    sum->pop_front(1);
}

// Transpose a tiled row [1,row_tiles] into a tiled column [row_tiles,1].
inline void transpose_tile_row_to_column(DataflowBuffer& in, DataflowBuffer& o, uint32_t row_tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(row_tiles);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);  // unary: in_id->srcA
    transpose_init(in_id);
    for (uint32_t block_start = 0; block_start < row_tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles =
            (row_tiles - block_start < max_dst_tiles) ? row_tiles - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            transpose_tile(in_id, block_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(row_tiles);
}

// Exact fp32 row sum for q/k L2 normalization. The shared SFPU reducer avoids four full-tile
// matrix multiplies and preserves the input for the caller-managed scratch lifetime.
inline void rowsum_k(uint32_t Mt, uint32_t Kt) {
    compute_kernel_lib::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dfb::scratch_one,
        dfb::ones,
        dfb::scratch_two,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(Mt, Kt));
}

// o[i] = rsqrt(in[i] + eps) [* scale]. in holds per-row sum-of-squares (rowsum_k output);
// out is the per-row inverse-L2 factor (optionally pre-scaled, for folding q's scale into the norm).
// eps/scale arrive as fp32-bit-cast uint32 compile args.
template <bool Scale>
inline void inverse_rms_tiles(
    DataflowBuffer& in, DataflowBuffer& o, uint32_t n, uint32_t eps_bits, uint32_t scale_bits) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
        }
        binop_with_scalar_tile_init();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            add_unary_tile(tile, eps_bits);
        }
        rsqrt_tile_init();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            rsqrt_tile(tile);
        }
        if constexpr (Scale) {
            binop_with_scalar_tile_init();
            for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                mul_unary_tile(tile, scale_bits);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

template <uint32_t RowTiles, uint32_t ColumnTiles, bool Scale>
inline void normalize_l2_rows(
    DataflowBuffer& input,
    DataflowBuffer& normalized,
    DataflowBuffer& squared,
    DataflowBuffer& row_sums,
    DataflowBuffer& inverse_norms,
    uint32_t eps_bits,
    uint32_t scale_bits) {
    constexpr uint32_t matrix_tiles = RowTiles * ColumnTiles;
    square_tiles(input, squared, matrix_tiles);
    rowsum_k(RowTiles, ColumnTiles);
    row_sums.wait_front(RowTiles);
    inverse_rms_tiles<Scale>(row_sums, inverse_norms, RowTiles, eps_bits, scale_bits);
    inverse_norms.wait_front(RowTiles);
    row_sums.pop_front(RowTiles);
    multiply_by_column(input, inverse_norms, normalized, RowTiles, ColumnTiles);
    normalized.wait_front(matrix_tiles);
    inverse_norms.pop_front(RowTiles);
    input.pop_front(matrix_tiles);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t SCALE_BITS, uint32_t EPS_BITS>
TT_KERNEL void compute(uint32_t work_item_count) {
    static_assert(Ct == 1, "chunk KDA currently requires chunk_size=32");

    constexpr uint32_t chunk_matrix_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;

    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer state(dfb::state);
    DataflowBuffer decay(dfb::decay);
    DataflowBuffer decay_exp(dfb::decay_exp);
    DataflowBuffer decay_factor(dfb::decay_factor);
    DataflowBuffer lower_mask(dfb::lower_mask);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer k_beta(dfb::k_beta);
    DataflowBuffer kd(dfb::w);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer state_two(dfb::state_two);
    DataflowBuffer v_new(dfb::v_new);
    DataflowBuffer output_intermediate(dfb::output_intermediate);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch_one(dfb::scratch_one);
    DataflowBuffer scratch_two(dfb::scratch_two);
    DataflowBuffer scratch_three(dfb::scratch_three);
    DataflowBuffer state_three(dfb::state_three);

    compute_kernel_hw_startup(dfb::q, dfb::k, dfb::scratch_one);
    eye.wait_front(chunk_matrix_tiles);
    tril.wait_front(chunk_matrix_tiles);
    ones.wait_front(chunk_matrix_tiles);

    for (uint32_t work_item = 0; work_item < work_item_count; ++work_item) {
        q.wait_front(chunk_key_tiles);
        k.wait_front(chunk_key_tiles);
        v.wait_front(chunk_value_tiles);
        g.wait_front(chunk_key_tiles);
        beta.wait_front(Ct);

        normalize_l2_rows<Ct, Kt, true>(
            q, state_temporary, scratch_one, scratch_two, scratch_three, EPS_BITS, SCALE_BITS);
        normalize_l2_rows<Ct, Kt, false>(k, final_state, scratch_one, scratch_two, scratch_three, EPS_BITS, SCALE_BITS);

        // v_beta and k_beta.
        multiply_by_column(v, beta, v_beta, Ct, Vt);
        v.pop_front(chunk_value_tiles);
        multiply_by_column(final_state, beta, k_beta, Ct, Kt);
        k_beta.wait_front(chunk_key_tiles);
        beta.pop_front(Ct);

        // G = cumsum(g). Anchor the separable pairwise factors at G_last/2 so neither
        // exp(G-anchor) nor exp(anchor-G) spans the full chunk range. Their products are
        // unchanged, while realistic KDA gates no longer overflow exp(-G).
        matmul_blocks<Ct, Ct, Kt, false>(tril, g, decay);
        decay.wait_front(chunk_key_tiles);
        exponential_tiles(decay, decay_exp, chunk_key_tiles);  // exp(G), for scan-facing q_decay/kd
        decay_exp.wait_front(chunk_key_tiles);

        matmul_blocks<Ct, Ct, Kt, false>(ones, g, scratch_one);  // replicated G_last
        scratch_one.wait_front(chunk_key_tiles);
        g.pop_front(chunk_key_tiles);
        multiply_by_half(scratch_one, scratch_two, chunk_key_tiles);  // anchor = G_last/2
        scratch_two.wait_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Subtract>(decay, scratch_two, output_intermediate, chunk_key_tiles);
        output_intermediate.wait_front(chunk_key_tiles);
        decay.pop_front(chunk_key_tiles);
        exponential_tiles(output_intermediate, decay, chunk_key_tiles);
        decay.wait_front(chunk_key_tiles);
        negated_exponential_tiles(output_intermediate, decay_factor, chunk_key_tiles);
        decay_factor.wait_front(chunk_key_tiles);
        output_intermediate.pop_front(chunk_key_tiles);

        exponential_tiles(scratch_two, state_update, chunk_key_tiles);  // exp(G_last/2)
        state_update.wait_front(chunk_key_tiles);

        // Preserve exact scan-facing factors, and use anchored factors only for pairwise products.
        elementwise_binary<ElementwiseBinaryOp::Multiply>(state_temporary, decay_exp, q_decay, chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(state_temporary, decay, state_three, chunk_key_tiles);
        state_three.wait_front(chunk_key_tiles);  // q*exp(G-anchor)
        state_temporary.pop_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(k_beta, decay_exp, kd, chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(k_beta, decay, state_two, chunk_key_tiles);
        state_two.wait_front(chunk_key_tiles);  // beta*k*exp(G-anchor)
        k_beta.pop_front(chunk_key_tiles);
        decay.pop_front(chunk_key_tiles);
        decay_exp.pop_front(chunk_key_tiles);
        exponential_tiles(scratch_one, decay, chunk_key_tiles);  // exp(G_last), for state decay dl
        decay.wait_front(chunk_key_tiles);
        scratch_one.pop_front(chunk_key_tiles);
        scratch_two.pop_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(final_state, decay_factor, scratch_one, chunk_key_tiles);
        scratch_one.wait_front(chunk_key_tiles);  // k*exp(anchor-G)
        final_state.pop_front(chunk_key_tiles);
        decay_factor.pop_front(chunk_key_tiles);

        // Materialize both anchored pairwise products, then release state_two/state_three before
        // the doubling inverse reuses those CBs as private scratch. Only the masked Aqk is published to
        // writer-facing intra; publishing the raw matrix creates a second consumer race.
        matmul_blocks<Ct, Kt, Ct, true>(state_two, scratch_one, lower_mask);
        lower_mask.wait_front(chunk_matrix_tiles);  // raw beta*k_i*k_j*exp(G_i-G_j)
        state_two.pop_front(chunk_key_tiles);
        matmul_blocks<Ct, Kt, Ct, true>(state_three, scratch_one, scratch_two);
        scratch_two.wait_front(chunk_matrix_tiles);  // raw q_i*k_j*exp(G_i-G_j)
        state_three.pop_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(scratch_two, tril, intra, chunk_matrix_tiles);
        scratch_two.pop_front(chunk_matrix_tiles);

        // T_inv = (I + strictly_lower(Akk))^-1.
        elementwise_binary<ElementwiseBinaryOp::Multiply>(lower_mask, tril, scratch_two, chunk_matrix_tiles);
        scratch_two.wait_front(chunk_matrix_tiles);  // lower(A), including diagonal
        lower_mask.pop_front(chunk_matrix_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(scratch_two, eye, scratch_three, chunk_matrix_tiles);
        scratch_three.wait_front(chunk_matrix_tiles);  // diag(A)
        elementwise_binary<ElementwiseBinaryOp::Subtract>(scratch_three, scratch_two, lower_mask, chunk_matrix_tiles);
        lower_mask.wait_front(chunk_matrix_tiles);  // -strictly_lower(A)
        scratch_three.pop_front(chunk_matrix_tiles);
        scratch_two.pop_front(chunk_matrix_tiles);
        invert_doubling(lower_mask, 0, t_inv, state, final_state, state_two, state_three, eye);
        lower_mask.pop_front(chunk_matrix_tiles);

        // k_dec_t = (kr * exp(G_last))^T.
        elementwise_binary<ElementwiseBinaryOp::Multiply>(scratch_one, state_update, scratch_two, chunk_key_tiles);
        scratch_two.wait_front(chunk_key_tiles);
        scratch_one.pop_front(chunk_key_tiles);
        transpose_tile_row_to_column(scratch_two, k_decay_transposed, Kt);
        scratch_two.pop_front(chunk_key_tiles);

        // dl [K,1] is the transpose of any replicated exp(G_last) row.
        transpose_tile_row_to_column(decay, v_new, Kt);
        state_update.pop_front(chunk_key_tiles);
        decay.pop_front(chunk_key_tiles);
        // v_beta, kd, q_decay, intra, k_dec_t, dl, T_inv stay pushed for the writer.
    }
}
