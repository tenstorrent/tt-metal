// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
//
// Chunk-parallel KDA prep. For cumulative vector gate G [C,K], factor
// exp(G_i-G_j) inside each q/k dot as exp(G_i-a)*exp(a-G_j), anchored at a = G_last/2:
//   qd=q*exp(G), kl=k*exp(G), kr=k*exp(a-G)
//   Akk=strictly_lower((beta*k*exp(G-a))@kr^T), Aqk=tril((q*exp(G-a))@kr^T)
//   kd=beta*kl, k_dec_t=(k*exp(G_last-G))^T, dl=expm1(G_last)=exp(G_last)-1 (complement-form decay).
// Every exponent (G = tril@g, the suffix sum G_last-G = strict_upper@g, G_last, a, G-a) is formed and consumed in DST
// (FP32). Passed through FP32->TF32 source registers they lost about one TF32 ulp of their magnitude: k_dec_t, then
// k*exp(a-G)*exp(a), came out biased low even at the chunk end where exp(G_last-G) = 1 exactly (-0.7% .. -2.1% per
// head at |G_last| 22 .. 87), which contracted the recurrent state (tt_metal_tracker-g1b.4.18).
// T_inv uses a face-blocked polynomial inverse so large gate magnitudes remain stable.
//
// Scalar decay (GDN): g is one log decay per token, a [C,1] column. The pairwise decay is formed in difference
// form and masked before the exp: D = tril @ (strict_lower * g) gives D[i,j] = sum_{j<t<=i} g_t below the diagonal
// and exactly 0 on and above it, so E = tril * exp(D) has no exponent above 0 at any decay. D is a sum over the pair's
// own tokens, never a difference of large cumulative sums, so it keeps g's relative precision even when |G_last| is
// large (tt_metal_tracker-g1b.5.4.2). Then Akk = (beta*k @ k^T) * E, Aqk = tril(q @ k^T) * E,
// q_decay/kd scale rows by exp(G) (G = tril @ g), k_dec_t scales columns by exp(G_last - G) = exp(ones @ (strict_lower
// * g)) <= 1, and dl = expm1(ones @ g) is replicated over the K rows.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/compute_kernel_api.h"  // expm1_tile
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

enum class ElementwiseBinaryOp { Add, Subtract, Multiply };

// Compute out[Mt,Nt] = A[Mt,Kt] @ (transpose_b ? B[Nt,Kt]^T : B[Kt,Nt]) in the largest rectangular
// subblocks that exactly divide the output and fit in destination registers.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr>
inline void matmul_blocks(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o) {
    constexpr uint32_t subblock_columns = kda::MatmulSubblock<Mt, Nt>::columns;
    constexpr uint32_t subblock_rows = kda::MatmulSubblock<Mt, Nt>::rows;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
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
        if constexpr (Op == ElementwiseBinaryOp::Add) {
            add_block(a_id, b_id, block_start, block_start, 0, block_tiles);
        } else if constexpr (Op == ElementwiseBinaryOp::Subtract) {
            sub_block(a_id, b_id, block_start, block_start, 0, block_tiles);
        } else {
            mul_block(a_id, b_id, block_start, block_start, 0, block_tiles);
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

// Multiply one selected source-tile pair and publish the result.
inline void multiply_selected_tile(
    DataflowBuffer& a, uint32_t a_tile, DataflowBuffer& b, uint32_t b_tile, DataflowBuffer& o) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(1);
    reconfig_data_format(a_id, b_id);
    mul_init(a_id, b_id);
    tile_regs_acquire();
    mul_block(a_id, b_id, a_tile, b_tile, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o_id, 0);
    tile_regs_release();
    o.push_back(1);
}

inline void square_tiles(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
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

// o[Ct,Kt] = exp(mask @ g) with the product kept in DST (FP32) through the exp, so the exponent never passes through a
// source register.
template <uint32_t Kt>
inline void exponential_of_mask_products(DataflowBuffer& mask, DataflowBuffer& g, DataflowBuffer& o) {
    const uint32_t mask_id = mask.get_id();
    const uint32_t g_id = g.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Kt);
    for (uint32_t block_start = 0; block_start < Kt; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (Kt - block_start < max_dst_tiles) ? Kt - block_start : max_dst_tiles;
        tile_regs_acquire();
        reconfig_data_format(g_id, mask_id);  // matmul: mask->srcB, g->srcA
        matmul_block_init(mask_id, g_id, false, 1, 1, 1);
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            matmul_block(mask_id, g_id, 0, block_start + tile, tile, false, 1, 1, 1);
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
    o.push_back(Kt);
}

// Complement-form decay: final_decay carries expm1(G_last) = exp(G_last) - 1 instead of exp(G_last). For
// long-memory channels exp(G_last) lies within 2^-9 of 1, where BF16 rounds it to exactly 1.0 and the channel
// stops forgetting; the small complement keeps its relative precision. recurrent_chunk_scan applies
// S + (S * final_decay + update).
//
// Per key-channel tile of one chunk, entirely in DST (FP32): G = tril @ g and the suffix sum S = strict_upper @ g =
// G_last - G, G_last = G + S, the anchor G_last/2 and the centered exponent c = G - anchor. Publishes exp(c),
// exp(-c), expm1(G_last) rows and exp(S). Four DST tiles.
template <uint32_t Kt>
inline void gate_factor_tiles(
    DataflowBuffer& tril,
    DataflowBuffer& strict_upper,
    DataflowBuffer& g,
    DataflowBuffer& centered_decay,
    DataflowBuffer& centered_inverse_decay,
    DataflowBuffer& final_decay_rows,
    DataflowBuffer& key_suffix_decay) {
    static_assert(max_dst_tiles >= 4, "gate factors need four DST tiles");
    constexpr uint32_t fp32_half_bits = __builtin_bit_cast(uint32_t, 0.5F);
    constexpr uint32_t fp32_two_bits = __builtin_bit_cast(uint32_t, 2.0F);
    const uint32_t tril_id = tril.get_id();
    const uint32_t upper_id = strict_upper.get_id();
    const uint32_t g_id = g.get_id();

    centered_decay.reserve_back(Kt);
    centered_inverse_decay.reserve_back(Kt);
    final_decay_rows.reserve_back(Kt);
    key_suffix_decay.reserve_back(Kt);
    for (uint32_t ki = 0; ki < Kt; ++ki) {
        tile_regs_acquire();
        reconfig_data_format(g_id, tril_id);  // matmul: masks->srcB, g->srcA
        matmul_block_init(tril_id, g_id, false, 1, 1, 1);
        matmul_block(tril_id, g_id, 0, ki, 0, false, 1, 1, 1);   // DST0 = G
        matmul_block(upper_id, g_id, 0, ki, 1, false, 1, 1, 1);  // DST1 = S = G_last - G
        add_binary_tile_init();
        add_binary_tile(0, 1, 2);  // DST2 = G_last
        binop_with_scalar_tile_init();
        mul_unary_tile(2, fp32_half_bits);  // DST2 = anchor
        sub_binary_tile_init();
        sub_binary_tile(0, 2, 3);  // DST3 = c = G - anchor
        sub_binary_tile(2, 0, 0);  // DST0 = -c
        binop_with_scalar_tile_init();
        mul_unary_tile(2, fp32_two_bits);  // DST2 = G_last (exact)
        expm1_tile_init<false>();
        expm1_tile<false>(2);
        exp_tile_init();
        exp_tile(3);
        exp_tile(0);
        exp_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(3, centered_decay.get_id(), ki);
        pack_tile(0, centered_inverse_decay.get_id(), ki);
        pack_tile(2, final_decay_rows.get_id(), ki);
        pack_tile(1, key_suffix_decay.get_id(), ki);
        tile_regs_release();
    }
    centered_decay.push_back(Kt);
    centered_inverse_decay.push_back(Kt);
    final_decay_rows.push_back(Kt);
    key_suffix_decay.push_back(Kt);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void multiply_by_column(DataflowBuffer& a, DataflowBuffer& col, DataflowBuffer& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t col_id = col.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
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

// Invert (I-N) for the strictly lower N by nesting three block levels, so every matmul keeps
// one strictly-lower factor and no large cancelling power of PS4 is ever formed.
//   level 1: 4-row diagonal blocks. N1^4=0, so A = (I-N1)^-1 = I+N1(I+N1(I+N1)) in two matmuls.
//   level 2: pair those blocks. M2 = A L2 satisfies M2^2=0, so B = (I+M2)A = A + M2 A.
//   level 3: join the 8-row blocks. M3 = B L3 satisfies M3^4=0, so
//            (I-N)^-1 = (I+M3)(I+M3^2)B.
// Eight matmuls in total, two fewer than a single 8-row Horner series over the same input.
inline void invert_block_nested(
    DataflowBuffer& negative_strict_lower_akk,
    DataflowBuffer& inverse,
    DataflowBuffer& identity,
    DataflowBuffer& block_masks,
    DataflowBuffer& matrix,
    DataflowBuffer& total,
    DataflowBuffer& product) {
    multiply_selected_tile(negative_strict_lower_akk, 0, block_masks, 0, matrix);  // N1
    matrix.wait_front(1);
    elementwise_binary<ElementwiseBinaryOp::Add>(identity, matrix, total, 1);
    total.wait_front(1);
    for (uint32_t step = 0; step < 2; ++step) {
        matmul_blocks<1, 1, 1, false>(matrix, total, product);
        product.wait_front(1);
        // The two-entry total DFB holds the old and new Horner values together.
        // Discarding the old front makes the new value current without a second DFB.
        elementwise_binary<ElementwiseBinaryOp::Add>(identity, product, total, 1);
        total.wait_front(2);
        total.pop_front(1);
        product.pop_front(1);
    }
    matrix.pop_front(1);  // total holds A

    multiply_selected_tile(negative_strict_lower_akk, 0, block_masks, 1, matrix);  // L2
    matrix.wait_front(1);
    matmul_blocks<1, 1, 1, false>(total, matrix, product);  // M2 = A L2
    product.wait_front(1);
    matrix.pop_front(1);
    matmul_blocks<1, 1, 1, false>(product, total, matrix);  // M2 A
    matrix.wait_front(1);
    product.pop_front(1);
    elementwise_binary<ElementwiseBinaryOp::Add>(total, matrix, total, 1);  // B = A + M2 A
    total.wait_front(2);
    total.pop_front(1);
    matrix.pop_front(1);

    multiply_selected_tile(negative_strict_lower_akk, 0, block_masks, 2, product);  // L3
    product.wait_front(1);
    // -strict_lower(Akk) is dead after extracting L3. Reuse its one-tile DFB
    // below for I+M3; all transactions remain one tile wide.
    negative_strict_lower_akk.pop_front(1);
    matmul_blocks<1, 1, 1, false>(total, product, matrix);  // M3 = B L3
    matrix.wait_front(1);
    product.pop_front(1);
    matmul_blocks<1, 1, 1, false>(matrix, matrix, product);  // M3^2
    product.wait_front(1);
    elementwise_binary<ElementwiseBinaryOp::Add>(identity, matrix, negative_strict_lower_akk, 1);
    negative_strict_lower_akk.wait_front(1);
    matrix.pop_front(1);
    elementwise_binary<ElementwiseBinaryOp::Add>(identity, product, matrix, 1);
    matrix.wait_front(1);
    product.pop_front(1);
    matmul_blocks<1, 1, 1, false>(negative_strict_lower_akk, matrix, product);  // (I+M3)(I+M3^2)
    product.wait_front(1);
    negative_strict_lower_akk.pop_front(1);
    matrix.pop_front(1);
    matmul_blocks<1, 1, 1, false>(product, total, inverse);
    product.pop_front(1);
    total.pop_front(1);
}

// Transpose a tiled row [1,row_tiles] into a tiled column [row_tiles,1].
inline void transpose_tile_row_to_column(DataflowBuffer& in, DataflowBuffer& o, uint32_t row_tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(row_tiles);
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

// Reduce each row of squared values directly to its inverse-L2 factor. Applying epsilon, rsqrt,
// and optional scaling in DST avoids materializing row sums in a separate intermediate DFB.
template <bool Scale, uint32_t SquaredDfb, uint32_t InverseNormsDfb>
inline void reduce_squared_rows_to_inverse_norms(
    uint32_t row_tiles, uint32_t column_tiles, uint32_t eps_bits, uint32_t scale_bits) {
    auto inverse_norm = [eps_bits, scale_bits](uint32_t dst) {
        binop_with_scalar_tile_init();
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
        if constexpr (Scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(dst, scale_bits);
        }
    };

    compute_kernel_lib::
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, SquaredDfb, dfb::ones, InverseNormsDfb>(
            compute_kernel_lib::ReduceInputBlockShape::of(row_tiles, column_tiles),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            inverse_norm);
}

template <uint32_t RowTiles, uint32_t ColumnTiles, bool Scale, uint32_t SquaredDfb, uint32_t InverseNormsDfb>
inline void normalize_l2_rows(
    DataflowBuffer& input,
    DataflowBuffer& normalized,
    uint32_t eps_bits,
    uint32_t scale_bits,

    // intermediate
    DataflowBuffer& squared,
    DataflowBuffer& inverse_norms) {
    constexpr uint32_t matrix_tiles = RowTiles * ColumnTiles;

    square_tiles(input, squared, matrix_tiles);

    reduce_squared_rows_to_inverse_norms<Scale, SquaredDfb, InverseNormsDfb>(
        RowTiles, ColumnTiles, eps_bits, scale_bits);

    inverse_norms.wait_front(RowTiles);
    multiply_by_column(input, inverse_norms, normalized, RowTiles, ColumnTiles);
    normalized.wait_front(matrix_tiles);
    inverse_norms.pop_front(RowTiles);
    input.pop_front(matrix_tiles);
}

template <uint32_t Ct, uint32_t Vt>
inline void prepare_v_beta(DataflowBuffer& v, DataflowBuffer& beta, DataflowBuffer& v_beta) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    multiply_by_column(v, beta, v_beta, Ct, Vt);
    v.pop_front(chunk_value_tiles);
}

template <uint32_t Ct, uint32_t Kt>
inline void prepare_gate_factors(
    DataflowBuffer& g,
    DataflowBuffer& prefix_sum_mask,
    DataflowBuffer& strict_upper,
    DataflowBuffer& decay,
    DataflowBuffer& centered_decay,
    DataflowBuffer& centered_inverse_decay,
    DataflowBuffer& final_decay_rows,
    DataflowBuffer& key_suffix_decay) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    // G = cumsum(g). Anchor the separable pairwise factors at G_last/2 so neither
    // exp(G-anchor) nor exp(anchor-G) spans the full chunk range. Their products are
    // unchanged, while realistic KDA gates no longer overflow exp(-G). All exponents stay in DST.
    exponential_of_mask_products<Kt>(prefix_sum_mask, g, decay);  // exp(G), for scan-facing q_decay/kd
    decay.wait_front(chunk_key_tiles);
    gate_factor_tiles<Kt>(
        prefix_sum_mask, strict_upper, g, centered_decay, centered_inverse_decay, final_decay_rows, key_suffix_decay);
    centered_decay.wait_front(chunk_key_tiles);
    centered_inverse_decay.wait_front(chunk_key_tiles);
    final_decay_rows.wait_front(chunk_key_tiles);
    key_suffix_decay.wait_front(chunk_key_tiles);
    g.pop_front(chunk_key_tiles);
}

template <uint32_t Ct, uint32_t Kt>
inline void prepare_scan_and_pairwise_inputs(
    DataflowBuffer& normalized_q,
    DataflowBuffer& normalized_k,
    DataflowBuffer& beta,
    DataflowBuffer& decay,
    DataflowBuffer& centered_decay,
    DataflowBuffer& q_decay,
    DataflowBuffer& kd,
    DataflowBuffer& k_beta_pairwise,
    DataflowBuffer& q_pairwise) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    {
        DataflowBuffer& beta_k = q_pairwise;
        multiply_by_column(normalized_k, beta, beta_k, Ct, Kt);
        beta_k.wait_front(chunk_key_tiles);
    }
    beta.pop_front(Ct);

    // Preserve exact scan-facing factors, and use anchored factors only for pairwise products.
    pack_reconfig_data_format(q_pairwise.get_id(), q_decay.get_id());
    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_q, decay, q_decay, chunk_key_tiles);
    {
        DataflowBuffer& beta_k = q_pairwise;
        pack_reconfig_data_format(q_decay.get_id(), kd.get_id());
        elementwise_binary<ElementwiseBinaryOp::Multiply>(beta_k, decay, kd, chunk_key_tiles);
        pack_reconfig_data_format(kd.get_id(), k_beta_pairwise.get_id());
        elementwise_binary<ElementwiseBinaryOp::Multiply>(beta_k, centered_decay, k_beta_pairwise, chunk_key_tiles);
        k_beta_pairwise.wait_front(chunk_key_tiles);  // beta*k*exp(G-anchor)
        beta_k.pop_front(chunk_key_tiles);
    }
    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_q, centered_decay, q_pairwise, chunk_key_tiles);
    q_pairwise.wait_front(chunk_key_tiles);  // q*exp(G-anchor)
    normalized_q.pop_front(chunk_key_tiles);
    centered_decay.pop_front(chunk_key_tiles);
    decay.pop_front(chunk_key_tiles);
}

// dl [K,1] is the transpose of any replicated expm1(G_last) row. Enters and leaves PACK in FP32.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_final_decay(DataflowBuffer& final_decay_rows, DataflowBuffer& final_decay) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    pack_reconfig_data_format(final_decay_rows.get_id(), final_decay.get_id());
    transpose_tile_row_to_column(final_decay_rows, final_decay, Kt);
    final_decay_rows.pop_front(chunk_key_tiles);
    pack_reconfig_data_format(final_decay.get_id(), final_decay_rows.get_id());
}

// k_dec = k * exp(G_last - G) (scan-facing) and the anchored k_pairwise = k * exp(anchor - G).
template <uint32_t Ct, uint32_t Kt>
inline void prepare_key_decays(
    DataflowBuffer& normalized_k,
    DataflowBuffer& key_suffix_decay,
    DataflowBuffer& centered_inverse_decay,
    DataflowBuffer& k_dec,
    DataflowBuffer& k_pairwise) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_k, key_suffix_decay, k_dec, chunk_key_tiles);
    k_dec.wait_front(chunk_key_tiles);
    key_suffix_decay.pop_front(chunk_key_tiles);
    elementwise_binary<ElementwiseBinaryOp::Multiply>(
        normalized_k, centered_inverse_decay, k_pairwise, chunk_key_tiles);
    k_pairwise.wait_front(chunk_key_tiles);  // k*exp(anchor-G)
    normalized_k.pop_front(chunk_key_tiles);
    centered_inverse_decay.pop_front(chunk_key_tiles);
}

template <uint32_t Ct, uint32_t Kt>
inline void prepare_pairwise_matrices(
    DataflowBuffer& k_beta_pairwise,
    DataflowBuffer& q_pairwise,
    DataflowBuffer& k_pairwise,
    DataflowBuffer& causal_mask,
    DataflowBuffer& akk,
    DataflowBuffer& intra,
    DataflowBuffer& aqk) {
    constexpr uint32_t chunk_matrix_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    // Materialize both anchored pairwise products, then release k_beta_pairwise/q_pairwise before
    // their whole-row storage is reused. Aqk uses separate single-tile scratch; only its masked value reaches
    // writer-facing intra; publishing the raw matrix creates a second consumer race.
    matmul_blocks<Ct, Kt, Ct, true>(k_beta_pairwise, k_pairwise, akk);
    akk.wait_front(chunk_matrix_tiles);  // raw beta*k_i*k_j*exp(G_i-G_j)
    k_beta_pairwise.pop_front(chunk_key_tiles);

    {
        matmul_blocks<Ct, Kt, Ct, true>(q_pairwise, k_pairwise, aqk);
        aqk.wait_front(chunk_matrix_tiles);  // raw q_i*k_j*exp(G_i-G_j)
        q_pairwise.pop_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(aqk, causal_mask, intra, chunk_matrix_tiles);
        aqk.pop_front(chunk_matrix_tiles);
    }
}

template <uint32_t Ct>
inline void prepare_t_inv(
    DataflowBuffer& akk,
    DataflowBuffer& causal_mask,
    DataflowBuffer& identity,
    DataflowBuffer& block_masks,
    DataflowBuffer& t_inv,

    // intermediate
    DataflowBuffer& scratch_0,
    DataflowBuffer& scratch_1,
    DataflowBuffer& product) {
    constexpr uint32_t chunk_matrix_tiles = Ct * Ct;

    {
        DataflowBuffer& lower_akk = scratch_0;
        DataflowBuffer& diagonal_akk = scratch_1;

        // T_inv = (I + strictly_lower(Akk))^-1.
        elementwise_binary<ElementwiseBinaryOp::Multiply>(akk, causal_mask, lower_akk, chunk_matrix_tiles);
        lower_akk.wait_front(chunk_matrix_tiles);  // lower(A), including diagonal
        akk.pop_front(chunk_matrix_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(lower_akk, identity, diagonal_akk, chunk_matrix_tiles);
        diagonal_akk.wait_front(chunk_matrix_tiles);  // diag(A)
        elementwise_binary<ElementwiseBinaryOp::Subtract>(diagonal_akk, lower_akk, akk, chunk_matrix_tiles);
        akk.wait_front(chunk_matrix_tiles);  // -strictly_lower(A)
        diagonal_akk.pop_front(chunk_matrix_tiles);
        lower_akk.pop_front(chunk_matrix_tiles);
    }

    invert_block_nested(
        akk,
        t_inv,
        identity,
        block_masks,
        /*matrix=*/scratch_0,
        /*total=*/scratch_1,
        /*product=*/product);
}

// k_dec_t = k_dec^T. PACK must be in k_dec_t's format.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_k_dec_t(DataflowBuffer& k_dec, DataflowBuffer& k_dec_t) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    transpose_tile_row_to_column(k_dec, k_dec_t, Kt);
    k_dec.pop_front(chunk_key_tiles);
}

// out[Mt,Nt] = A[Mt,Nt] * row[0,:]  (broadcast the first row of the single `row` tile down every A tile)
inline void multiply_by_row(DataflowBuffer& a, DataflowBuffer& row, DataflowBuffer& o, uint32_t n) {
    const uint32_t a_id = a.get_id();
    const uint32_t row_id = row.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_data_format(a_id, row_id);  // bcast(a_id,row_id): a_id->srcA, row_id->srcB
    mul_bcast_rows_init(a_id, row_id);
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            mul_tiles_bcast_rows(a_id, row_id, block_start + tile, 0, tile);
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

enum class ProductExponential { Exp, Expm1 };

// o = op(A @ B) for single tiles A and B, the product kept in DST (FP32 accumulation) through the SFPU op. The
// result tile is packed `copies` times (a replicated column feeds every K row of dl).
template <ProductExponential Op>
inline void exponential_of_product(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o, uint32_t copies) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(copies);
    reconfig_data_format(b_id, a_id);  // matmul: a_id->srcB, b_id->srcA
    matmul_block_init(a_id, b_id, false, 1, 1, 1);
    tile_regs_acquire();
    matmul_block(a_id, b_id, 0, 0, 0, false, 1, 1, 1);
    if constexpr (Op == ProductExponential::Exp) {
        exp_tile_init();
        exp_tile(0);
    } else {
        expm1_tile_init<false>();
        expm1_tile<false>(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t copy = 0; copy < copies; ++copy) {
        pack_tile(0, o_id, copy);
    }
    tile_regs_release();
    o.push_back(copies);
}

// o = (left[1,Kt] @ right[1,Kt]^T) * tril * exp(tril @ masked_gate), one DST pass: the product accumulates in DST0,
// the masked difference D = tril @ masked_gate lands in DST1, and the SFPU applies exp, the causal mask, and the
// product there, so neither D nor the pairwise decay passes through a source register (in the style of the
// chunk_gated_delta_rule op's lmask_fused/negn_fused).
template <uint32_t Kt>
inline void causal_decayed_product(
    DataflowBuffer& left, DataflowBuffer& right, DataflowBuffer& tril, DataflowBuffer& masked_gate, DataflowBuffer& o) {
    const uint32_t left_id = left.get_id();
    const uint32_t right_id = right.get_id();
    const uint32_t tril_id = tril.get_id();
    const uint32_t gate_id = masked_gate.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format(right_id, left_id);  // matmul: left->srcB, right->srcA
    matmul_block_init(left_id, right_id, true, 1, 1, Kt);
    for (uint32_t ki = 0; ki < Kt; ++ki) {
        matmul_block(left_id, right_id, ki, ki, 0, true, 1, 1, Kt);  // DST0 = left @ right^T
    }
    reconfig_data_format(gate_id, tril_id);
    matmul_block_init(tril_id, gate_id, false, 1, 1, 1);
    matmul_block(tril_id, gate_id, 0, 0, 1, false, 1, 1, 1);  // DST1 = D, zero on and above the diagonal
    reconfig_data_format_srca(tril_id);
    copy_init(tril_id);
    copy_tile(tril_id, 0, 2);  // DST2 = tril (0/1, exact through SrcA)
    exp_tile_init();
    exp_tile(1);
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);  // E = tril * exp(D)
    mul_binary_tile(0, 1, 0);  // product * E
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o_id, 0);
    tile_regs_release();
    o.push_back(1);
}

// Scalar-decay gate terms for one (head, chunk): masked gate, exp(G) column, dl = expm1(G_last) over K rows, and
// exp(G_last - G) as a row. Consumes g. PACK enters and leaves in FP32 except for the final_decay transition.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_scalar_gate_terms(
    DataflowBuffer& g,
    DataflowBuffer& strict_lower,
    DataflowBuffer& tril,
    DataflowBuffer& ones,
    DataflowBuffer& masked_gate,
    DataflowBuffer& cumulative_decay,
    DataflowBuffer& final_decay,
    DataflowBuffer& suffix_decay) {
    multiply_by_column(strict_lower, g, masked_gate, Ct, Ct);  // X[t,j] = g_t for t > j
    masked_gate.wait_front(Ct);
    exponential_of_product<ProductExponential::Exp>(tril, g, cumulative_decay, Ct);  // exp(G), G = cumsum(g)
    pack_reconfig_data_format(cumulative_decay.get_id(), final_decay.get_id());
    exponential_of_product<ProductExponential::Expm1>(ones, g, final_decay, Kt);  // expm1(G_last) in every row
    g.pop_front(Ct);
    pack_reconfig_data_format(final_decay.get_id(), suffix_decay.get_id());
    // ones @ X: every row holds sum_{t>j} g_t = G_last - G_j, a sum over the suffix (no cancellation).
    exponential_of_product<ProductExponential::Exp>(ones, masked_gate, suffix_decay, Ct);
    cumulative_decay.wait_front(Ct);
    suffix_decay.wait_front(Ct);
}

// Scalar decay mode for one (head, chunk), after Q/K normalization and v_beta. See the file header.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_scalar_decay_chunk(
    DataflowBuffer& g,
    DataflowBuffer& beta,
    DataflowBuffer& normalized_q,
    DataflowBuffer& normalized_k,
    DataflowBuffer& strict_lower,
    DataflowBuffer& tril,
    DataflowBuffer& ones,
    DataflowBuffer& identity,
    DataflowBuffer& block_masks,
    DataflowBuffer& masked_gate,
    DataflowBuffer& cumulative_decay,
    DataflowBuffer& suffix_decay,
    DataflowBuffer& kd,
    DataflowBuffer& q_decay,
    DataflowBuffer& intra,
    DataflowBuffer& k_dec_t,
    DataflowBuffer& final_decay,
    DataflowBuffer& t_inv,
    DataflowBuffer& akk,

    // intermediate
    DataflowBuffer& beta_k,
    DataflowBuffer& transposed_k,
    DataflowBuffer& scratch_0,
    DataflowBuffer& scratch_1,
    DataflowBuffer& product) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    prepare_scalar_gate_terms<Ct, Kt>(
        g, strict_lower, tril, ones, masked_gate, cumulative_decay, final_decay, suffix_decay);

    multiply_by_column(normalized_k, beta, beta_k, Ct, Kt);
    beta_k.wait_front(chunk_key_tiles);
    beta.pop_front(Ct);
    pack_reconfig_data_format(beta_k.get_id(), kd.get_id());
    multiply_by_column(beta_k, cumulative_decay, kd, Ct, Kt);
    pack_reconfig_data_format(kd.get_id(), q_decay.get_id());
    multiply_by_column(normalized_q, cumulative_decay, q_decay, Ct, Kt);
    cumulative_decay.pop_front(Ct);

    pack_reconfig_data_format(q_decay.get_id(), akk.get_id());
    causal_decayed_product<Kt>(beta_k, normalized_k, tril, masked_gate, akk);  // beta*k_i*k_j*E_ij, E lower
    akk.wait_front(Ct * Ct);
    beta_k.pop_front(chunk_key_tiles);
    causal_decayed_product<Kt>(normalized_q, normalized_k, tril, masked_gate, intra);  // tril(q_i*k_j*E_ij)
    normalized_q.pop_front(chunk_key_tiles);
    masked_gate.pop_front(Ct);

    // k_dec_t = k^T * exp(G_last - G) along the chunk (columns of k^T).
    transpose_tile_row_to_column(normalized_k, transposed_k, Kt);
    transposed_k.wait_front(chunk_key_tiles);
    normalized_k.pop_front(chunk_key_tiles);
    pack_reconfig_data_format(transposed_k.get_id(), k_dec_t.get_id());
    multiply_by_row(transposed_k, suffix_decay, k_dec_t, chunk_key_tiles);
    transposed_k.pop_front(chunk_key_tiles);
    suffix_decay.pop_front(Ct);
    pack_reconfig_data_format(k_dec_t.get_id(), scratch_0.get_id());

    prepare_t_inv<Ct>(akk, tril, identity, block_masks, t_inv, scratch_0, scratch_1, product);
}

// Waits for one work item's inputs, normalizes Q and K, and emits v_beta. PACK leaves in v_beta's format.
// FORCE_INLINE: a lambda here cost the per-channel production case 0.75% of device time (code placement).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t GateTiles, uint32_t SCALE_BITS, uint32_t EPS_BITS>
FORCE_INLINE void prepare_normalized_inputs(
    DataflowBuffer& q,
    DataflowBuffer& k,
    DataflowBuffer& v,
    DataflowBuffer& g,
    DataflowBuffer& beta,
    DataflowBuffer& normalized_q,
    DataflowBuffer& normalized_k,
    DataflowBuffer& v_beta,

    // intermediate
    DataflowBuffer& squared,
    DataflowBuffer& inverse_norms) {
    q.wait_front(Ct * Kt);
    k.wait_front(Ct * Kt);
    v.wait_front(Ct * Vt);
    g.wait_front(GateTiles);
    beta.wait_front(Ct);

    normalize_l2_rows<Ct, Kt, true, dfb::workspace_3, dfb::tile_workspace_0>(
        q, normalized_q, EPS_BITS, SCALE_BITS, squared, inverse_norms);
    normalize_l2_rows<Ct, Kt, false, dfb::workspace_3, dfb::tile_workspace_0>(
        k, normalized_k, EPS_BITS, SCALE_BITS, squared, inverse_norms);

    // PACK state persists across helpers. Reconfigure only at actual destination-format transitions.
    pack_reconfig_data_format(normalized_k.get_id(), v_beta.get_id());
    prepare_v_beta<Ct, Vt>(v, beta, v_beta);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t SCALE_BITS, uint32_t EPS_BITS, uint32_t scalar_decay>
TT_KERNEL void compute(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks) {
    DataflowBuffer control(dfb::chronology_compute);
    const uint32_t valid_chunks = kda_chronology::receive(control).valid_rows / tt::constants::TILE_HEIGHT;
    static_assert(Ct == 1, "chunk KDA currently requires chunk_size=32");

    constexpr uint32_t chunk_matrix_tiles = Ct * Ct;
    constexpr uint32_t chunk_gate_tiles = scalar_decay ? Ct : Ct * Kt;

    // Reader-produced inputs and constants.
    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer block_masks(dfb::block_masks);
    DataflowBuffer ones(dfb::ones);

    // Writer-consumed outputs.
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);

    // Semantic intermediates shared by both decay modes.
    DataflowBuffer normalized_q(dfb::normalized_q);
    DataflowBuffer normalized_k(dfb::normalized_k);
    DataflowBuffer tile_workspace_0(dfb::tile_workspace_0);
    DataflowBuffer tile_workspace_1(dfb::tile_workspace_1);
    DataflowBuffer tile_workspace_2(dfb::tile_workspace_2);

    DataflowBuffer akk(dfb::akk);

    // Reusable physical storage. Live values crossing helper boundaries are named below.
    DataflowBuffer workspace_1(dfb::workspace_1);
    DataflowBuffer workspace_3(dfb::workspace_3);

    compute_kernel_hw_startup(dfb::q, dfb::k, dfb::workspace_3);
    eye.wait_front(chunk_matrix_tiles);
    tril.wait_front(chunk_matrix_tiles);
    block_masks.wait_front(2);
    ones.wait_front(chunk_matrix_tiles);

    if constexpr (scalar_decay) {
        DataflowBuffer strict_lower(*dfb::get_token_if_present<"strict_lower">());
        DataflowBuffer masked_gate(*dfb::get_token_if_present<"masked_gate">());
        DataflowBuffer cumulative_decay(*dfb::get_token_if_present<"cumulative_decay">());
        DataflowBuffer suffix_decay(*dfb::get_token_if_present<"suffix_decay">());

        // strictly lower = tril - I, exact in FP32; held for the whole kernel.
        elementwise_binary<ElementwiseBinaryOp::Subtract>(tril, eye, strict_lower, chunk_matrix_tiles);
        strict_lower.wait_front(chunk_matrix_tiles);

        for (uint32_t work_item = 0; work_item < work_item_count; ++work_item) {
            if ((work_item_start + work_item) % num_chunks >= valid_chunks) {
                continue;
            }
            prepare_normalized_inputs<Ct, Kt, Vt, chunk_gate_tiles, SCALE_BITS, EPS_BITS>(
                q, k, v, g, beta, normalized_q, normalized_k, v_beta, workspace_3, tile_workspace_0);
            pack_reconfig_data_format(v_beta.get_id(), masked_gate.get_id());
            prepare_scalar_decay_chunk<Ct, Kt>(
                g,
                beta,
                normalized_q,
                normalized_k,
                strict_lower,
                tril,
                ones,
                eye,
                block_masks,
                masked_gate,
                cumulative_decay,
                suffix_decay,
                kd,
                q_decay,
                intra,
                k_decay_transposed,
                final_decay,
                t_inv,
                akk,
                /*beta_k=*/workspace_1,
                /*transposed_k=*/workspace_3,
                /*scratch_0=*/tile_workspace_0,
                /*scratch_1=*/tile_workspace_1,
                /*product=*/tile_workspace_2);
            // PACK leaves in FP32 (t_inv), the format the next item's first pack expects.
        }
    } else {
        DataflowBuffer scan_decay(*dfb::get_token_if_present<"scan_decay">());
        DataflowBuffer centered_inverse_decay(*dfb::get_token_if_present<"centered_inverse_decay">());
        DataflowBuffer key_suffix_decay(*dfb::get_token_if_present<"key_suffix_decay">());
        DataflowBuffer strict_upper(*dfb::get_token_if_present<"strict_upper">());
        DataflowBuffer workspace_0(*dfb::get_token_if_present<"workspace_0">());
        DataflowBuffer workspace_2(*dfb::get_token_if_present<"workspace_2">());

        // strictly upper = ones - tril, exact in FP32; held for the whole kernel.
        elementwise_binary<ElementwiseBinaryOp::Subtract>(ones, tril, strict_upper, chunk_matrix_tiles);
        strict_upper.wait_front(chunk_matrix_tiles);

        for (uint32_t work_item = 0; work_item < work_item_count; ++work_item) {
            if ((work_item_start + work_item) % num_chunks >= valid_chunks) {
                continue;
            }
            prepare_normalized_inputs<Ct, Kt, Vt, chunk_gate_tiles, SCALE_BITS, EPS_BITS>(
                q, k, v, g, beta, normalized_q, normalized_k, v_beta, workspace_3, tile_workspace_0);
            pack_reconfig_data_format(v_beta.get_id(), workspace_0.get_id());

            DataflowBuffer& centered_decay = workspace_0;
            DataflowBuffer& final_decay_rows = workspace_3;
            prepare_gate_factors<Ct, Kt>(
                g,
                tril,
                strict_upper,
                scan_decay,
                centered_decay,
                centered_inverse_decay,
                final_decay_rows,
                key_suffix_decay);

            DataflowBuffer& k_beta_pairwise = workspace_1;
            DataflowBuffer& q_pairwise = workspace_2;
            prepare_scan_and_pairwise_inputs<Ct, Kt>(
                normalized_q, normalized_k, beta, scan_decay, centered_decay, q_decay, kd, k_beta_pairwise, q_pairwise);

            prepare_final_decay<Ct, Kt>(final_decay_rows, final_decay);

            DataflowBuffer& k_dec = workspace_0;
            DataflowBuffer& k_pairwise = workspace_3;
            prepare_key_decays<Ct, Kt>(normalized_k, key_suffix_decay, centered_inverse_decay, k_dec, k_pairwise);

            prepare_pairwise_matrices<Ct, Kt>(
                k_beta_pairwise, q_pairwise, k_pairwise, tril, akk, intra, tile_workspace_0);
            k_pairwise.pop_front(Ct * Kt);

            // All inverse scratch transactions are one tile. tile_workspace_1 has two entries so each
            // level can enqueue its replacement before popping the old value; its four transactions
            // per work item also return both cursors to their starting slot.
            prepare_t_inv<Ct>(
                akk,
                tril,
                eye,
                block_masks,
                t_inv,
                /*scratch_0=*/tile_workspace_0,
                /*scratch_1=*/tile_workspace_1,
                /*product=*/tile_workspace_2);

            pack_reconfig_data_format(t_inv.get_id(), k_decay_transposed.get_id());
            prepare_k_dec_t<Ct, Kt>(k_dec, k_decay_transposed);
            pack_reconfig_data_format(k_decay_transposed.get_id(), workspace_3.get_id());
            // v_beta, kd, q_decay, intra, k_dec_t, dl, T_inv stay pushed for the writer.
        }
    }
}
