// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
//
// Chunk-parallel KDA prep. For cumulative vector gate G [C,K], factor
// exp(G_i-G_j) inside each q/k dot as exp(G_i)*exp(-G_j):
//   qd=q*exp(G), kl=k*exp(G), kr=k*exp(-G)
//   Akk=strictly_lower((beta*kl)@kr^T), Aqk=tril(qd@kr^T)
//   kd=beta*kl, k_dec_t=(kr*exp(G_last))^T, dl=exp(G_last).
// dl is logically [K,1]: only its column 0 is defined, and consumers broadcast that column.
// T_inv uses a face-blocked polynomial inverse so large gate magnitudes remain stable.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Code size: the whole program must fit the kernel config buffer. The pack build keeps the helpers out of line and
// its pack loops rolled (`#pragma GCC unroll 1`). The unpack and math builds declare them inline: out of line, the
// unpack thread's format reconfiguration cannot fold the buffer ids, and its binary grows.
#ifdef TRISC_PACK
#define KDA_PREP_HELPER __attribute__((noinline))
#else
#define KDA_PREP_HELPER inline
#endif

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

enum class ElementwiseBinaryOp { Add, Subtract, Multiply };

// Compute out[Mt,Nt] = A[Mt,Kt] @ (transpose_b ? B[Nt,Kt]^T : B[Kt,Nt]) in the largest rectangular
// subblocks that exactly divide the output and fit in destination registers.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void matmul_blocks(DFB0& a, DFB1& b, DFB2& o) {
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
#pragma GCC unroll 1
                for (uint32_t column = 0; column < subblock_columns; column++) {
                    pack_tile(row * subblock_columns + column, o_id, (mi + row) * Nt + ni + column);
                }
            }
            tile_regs_release();
        }
    }
    o.push_back(Mt * Nt);
}

// Tilewise binary operation, shared by every elementwise stage and by the batched inverse, where the core's work
// items run each stage together, up to max_dst_tiles items per destination-register acquire, so the inverse pays
// its ~20 dependent stages once per core rather than once per item.
// a's tile for item i is a_base + i (or a_base alone when a is a shared constant), likewise b. Out of line, with the
// operation a runtime argument: the program sits at the kernel-config size limit.
template <typename DFB0, typename DFB1, typename DFB2>
__attribute__((noinline)) void binary_items(
    ElementwiseBinaryOp op,
    DFB0& a,
    uint32_t a_base,
    bool a_shared,
    DFB1& b,
    uint32_t b_base,
    bool b_shared,
    DFB2& o,
    uint32_t n) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(n);
    reconfig_data_format(a_id, b_id);
    if (op == ElementwiseBinaryOp::Add) {
        add_init(a_id, b_id);
    } else if (op == ElementwiseBinaryOp::Subtract) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t first = 0; first < n; first += max_dst_tiles) {
        const uint32_t count = n - first < max_dst_tiles ? n - first : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t item = 0; item < count; ++item) {
            const uint32_t a_tile = a_shared ? a_base : a_base + first + item;
            const uint32_t b_tile = b_shared ? b_base : b_base + first + item;
            if (op == ElementwiseBinaryOp::Add) {
                add_block(a_id, b_id, a_tile, b_tile, item, 1);
            } else if (op == ElementwiseBinaryOp::Subtract) {
                sub_block(a_id, b_id, a_tile, b_tile, item, 1);
            } else {
                mul_block(a_id, b_id, a_tile, b_tile, item, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t item = 0; item < count; ++item) {
            pack_tile(item, o_id, first + item);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// Apply a binary operation tilewise to n tiles of a and b.
template <ElementwiseBinaryOp Op, typename DFB0, typename DFB1, typename DFB2>
inline void elementwise_binary(DFB0& a, DFB1& b, DFB2& o, uint32_t n) {
    binary_items(Op, a, 0, false, b, 0, false, o, n);
}

template <typename DFB0, typename DFB1>
KDA_PREP_HELPER void square_tiles(DFB0& in, DFB1& o, uint32_t n) {
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
#pragma GCC unroll 1
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// Mode and Iterations restrict the exponential to a row vector for row-replicated inputs: VectorMode::R visits faces 0
// and 1, and two of a face's eight SFPU iterations span its first rows, row 0 included. The rest of each output tile
// is then unspecified.
template <VectorMode Mode = VectorMode::RC, int Iterations = 8, typename DFB0, typename DFB1>
KDA_PREP_HELPER void exponential_tiles(DFB0& in, DFB1& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_data_format_srca(in_id);  // unary: in_id->srcA
    copy_init(in_id);
    exp_tile_init();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            exp_tile<false, false, InputClamping::ClampToNegative, Iterations>(tile, Mode);
        }
        tile_regs_commit();
        tile_regs_wait();
#pragma GCC unroll 1
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

template <typename DFB0, typename DFB1>
KDA_PREP_HELPER void reciprocal_tiles(DFB0& in, DFB1& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    // Precise regardless of the op's approx mode: the reciprocal of exp(G - a) stands in for the exact exp(a - G).
    constexpr auto recip_dest = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16;
    recip_tile_init<recip_dest, ReciprocalApproxMode::Precise>();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            recip_tile<recip_dest, ReciprocalApproxMode::Precise>(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
#pragma GCC unroll 1
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
template <typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void multiply_by_column(DFB0& a, DFB1& col, DFB2& o, uint32_t Mt, uint32_t Nt) {
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
#pragma GCC unroll 1
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(Mt * Nt);
}

// out[n] = A[n] * row[n]  (broadcast row 0 of each `row` tile down its tile)
template <typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void multiply_by_row(DFB0& a, DFB1& row, DFB2& o, uint32_t n) {
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
            mul_tiles_bcast_rows(a_id, row_id, block_start + tile, block_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
#pragma GCC unroll 1
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// o[i] = a[i] @ b[i] for one-tile matrices.
template <typename DFB0, typename DFB1, typename DFB2>
__attribute__((noinline)) void matmul_items(DFB0& a, DFB1& b, DFB2& o, uint32_t n) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(n);
    reconfig_data_format(b_id, a_id);
    matmul_block_init(a_id, b_id, false, 1, 1, 1);
    for (uint32_t first = 0; first < n; first += max_dst_tiles) {
        const uint32_t count = n - first < max_dst_tiles ? n - first : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t item = 0; item < count; ++item) {
            matmul_block(a_id, b_id, first + item, first + item, item, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t item = 0; item < count; ++item) {
            pack_tile(item, o_id, first + item);
        }
        tile_regs_release();
    }
    o.push_back(n);
}

// T_inv = (I + strictly_lower(Akk))^-1 for the n items whose raw Akk tiles are the front of `akk`.
// Invert (I-N) for the strictly lower N = -strictly_lower(Akk) by nesting three block levels, so every matmul keeps
// one strictly-lower factor and no large cancelling power of PS4 is ever formed:
//   level 1: 4-row diagonal blocks. N1^4=0, so A = (I-N1)^-1 = I+N1(I+N1(I+N1)) in two matmuls.
//   level 2: pair those blocks. M2 = A L2 satisfies M2^2=0, so B = (I+M2)A = A + M2 A.
//   level 3: join the 8-row blocks. M3 = B L3 satisfies M3^4=0, so (I-N)^-1 = (I+M3)(I+M3^2)B.
// The items are independent: each stage runs over all n of them, its tile for item i at the front of its buffers + i.
template <
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6,
    typename DFB7>
__attribute__((noinline)) void invert_items(
    DFB0& akk,
    DFB1& causal_mask,
    DFB2& identity,
    DFB3& block_masks,
    DFB4& inverse,
    DFB5& matrix,
    DFB6& total,
    DFB7& product,
    uint32_t n) {
    using Op = ElementwiseBinaryOp;
    akk.wait_front(n);
    {
        auto& lower = matrix;
        auto& diagonal = total;
        binary_items(Op::Multiply, akk, 0, false, causal_mask, 0, true, lower, n);  // lower(A), with diagonal
        lower.wait_front(n);
        akk.pop_front(n);
        binary_items(Op::Multiply, lower, 0, false, identity, 0, true, diagonal, n);  // diag(A)
        diagonal.wait_front(n);
        binary_items(Op::Subtract, diagonal, 0, false, lower, 0, false, akk, n);  // -strictly_lower(A)
        akk.wait_front(n);
        diagonal.pop_front(n);
        lower.pop_front(n);
    }
    auto& negative_strict_lower = akk;

    binary_items(Op::Multiply, negative_strict_lower, 0, false, block_masks, 0, true, matrix, n);  // N1
    matrix.wait_front(n);
    binary_items(Op::Add, identity, 0, true, matrix, 0, false, total, n);
    total.wait_front(n);
    for (uint32_t step = 0; step < 2; ++step) {
        matmul_items(matrix, total, product, n);
        product.wait_front(n);
        // The double-depth total holds the old and new Horner values together.
        binary_items(Op::Add, identity, 0, true, product, 0, false, total, n);
        total.wait_front(2 * n);
        total.pop_front(n);
        product.pop_front(n);
    }
    matrix.pop_front(n);  // total holds A

    binary_items(Op::Multiply, negative_strict_lower, 0, false, block_masks, 1, true, matrix, n);  // L2
    matrix.wait_front(n);
    matmul_items(total, matrix, product, n);  // M2 = A L2
    product.wait_front(n);
    matrix.pop_front(n);
    matmul_items(product, total, matrix, n);  // M2 A
    matrix.wait_front(n);
    product.pop_front(n);
    binary_items(Op::Add, total, 0, false, matrix, 0, false, total, n);  // B = A + M2 A
    total.wait_front(2 * n);
    total.pop_front(n);
    matrix.pop_front(n);

    binary_items(Op::Multiply, negative_strict_lower, 0, false, block_masks, 2, true, product, n);  // L3
    product.wait_front(n);
    // -strictly_lower(Akk) is dead after extracting L3; its buffer holds I+M3 below.
    negative_strict_lower.pop_front(n);
    matmul_items(total, product, matrix, n);  // M3 = B L3
    matrix.wait_front(n);
    product.pop_front(n);
    matmul_items(matrix, matrix, product, n);  // M3^2
    product.wait_front(n);
    binary_items(Op::Add, identity, 0, true, matrix, 0, false, negative_strict_lower, n);  // I+M3
    negative_strict_lower.wait_front(n);
    matrix.pop_front(n);
    binary_items(Op::Add, identity, 0, true, product, 0, false, matrix, n);  // I+M3^2
    matrix.wait_front(n);
    product.pop_front(n);
    matmul_items(negative_strict_lower, matrix, product, n);  // (I+M3)(I+M3^2)
    product.wait_front(n);
    negative_strict_lower.pop_front(n);
    matrix.pop_front(n);
    matmul_items(product, total, inverse, n);
    product.pop_front(n);
    total.pop_front(n);
}

// Transpose a tiled row [1,row_tiles] into a tiled column [row_tiles,1].
template <typename DFB0, typename DFB1>
KDA_PREP_HELPER void transpose_tile_row_to_column(DFB0& in, DFB1& o, uint32_t row_tiles) {
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
#pragma GCC unroll 1
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
KDA_PREP_HELPER void reduce_squared_rows_to_inverse_norms(
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

template <
    uint32_t RowTiles,
    uint32_t ColumnTiles,
    bool Scale,
    uint32_t SquaredDfb,
    uint32_t InverseNormsDfb,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3>
KDA_PREP_HELPER void normalize_l2_rows(
    DFB0& input,
    DFB1& normalized,
    uint32_t eps_bits,
    uint32_t scale_bits,

    // intermediate
    DFB2& squared,
    DFB3& inverse_norms) {
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

// beta = sigmoid(logits) in FP32, from the BF16 logits the reader moved into column 0. The SFPU sigmoid is the
// accurate one ttnn.sigmoid runs; BF16 widens exactly into the FP32 destination.
template <uint32_t Ct, typename DFB0, typename DFB1>
KDA_PREP_HELPER void activate_beta(DFB0& logits, DFB1& beta) {
    beta.reserve_back(Ct);
    reconfig_data_format_srca(logits.get_id());
    pack_reconfig_data_format(beta.get_id());
    copy_init(logits.get_id());
    sigmoid_tile_init<false>();
    for (uint32_t tile = 0; tile < Ct; ++tile) {
        tile_regs_acquire();
        copy_tile(logits.get_id(), tile, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, beta.get_id(), tile);
        tile_regs_release();
    }
    beta.push_back(Ct);
    logits.pop_front(Ct);
}

template <uint32_t Ct, uint32_t Vt, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void prepare_v_beta(DFB0& v, DFB1& beta, DFB2& v_beta) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    multiply_by_column(v, beta, v_beta, Ct, Vt);
    v.pop_front(chunk_value_tiles);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6>
KDA_PREP_HELPER void prepare_gate_factors(
    DFB0& g,
    DFB1& prefix_sum_mask,
    DFB2& sum_broadcast_matrix,
    DFB3& centered_decay,
    DFB4& centered_inverse_decay,
    DFB5& anchor_decay,

    // intermediate
    DFB6& anchor_g) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    // G = cumsum(g). Anchor the separable pairwise factors at G_last/2 so neither
    // exp(G-anchor) nor exp(anchor-G) spans the full chunk range: their products are
    // those of exp(G) and exp(-G), which realistic KDA gates would overflow.
    matmul_blocks<Ct, Ct, Kt, false>(prefix_sum_mask, g, centered_decay);
    centered_decay.wait_front(chunk_key_tiles);

    // The sum-broadcast matrix carries half the gate scale, so this is the anchor G_last/2, replicated.
    matmul_blocks<Ct, Ct, Kt, false>(sum_broadcast_matrix, g, anchor_g);
    anchor_g.wait_front(chunk_key_tiles);
    g.pop_front(chunk_key_tiles);

    {
        auto& centered_g = anchor_decay;
        elementwise_binary<ElementwiseBinaryOp::Subtract>(centered_decay, anchor_g, centered_g, chunk_key_tiles);
        centered_g.wait_front(chunk_key_tiles);
        centered_decay.pop_front(chunk_key_tiles);
        exponential_tiles(centered_g, centered_decay, chunk_key_tiles);
        centered_decay.wait_front(chunk_key_tiles);
        reciprocal_tiles(centered_decay, centered_inverse_decay, chunk_key_tiles);  // exp(anchor-G)
        centered_inverse_decay.wait_front(chunk_key_tiles);
        centered_g.pop_front(chunk_key_tiles);
    }

    // exp(G_last/2) is row-replicated and the broadcasts read only row 0, so only the face rows holding it are
    // computed. That takes two SFPU iterations: one leaves part of row 0 unwritten, and kd, q_decay, k_dec_t and
    // final_decay come out wrong.
    exponential_tiles<VectorMode::R, 2>(anchor_g, anchor_decay, chunk_key_tiles);
    anchor_decay.wait_front(chunk_key_tiles);
    anchor_g.pop_front(chunk_key_tiles);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6,
    typename DFB7,
    typename DFB8>
KDA_PREP_HELPER void prepare_scan_and_pairwise_inputs(
    DFB0& normalized_q,
    DFB1& normalized_k,
    DFB2& beta,
    DFB3& anchor_decay,
    DFB4& centered_decay,
    DFB5& q_decay,
    DFB6& kd,
    DFB7& k_beta_pairwise,
    DFB8& q_pairwise) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    {
        auto& beta_k = q_pairwise;
        multiply_by_column(normalized_k, beta, beta_k, Ct, Kt);
        beta_k.wait_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(beta_k, centered_decay, k_beta_pairwise, chunk_key_tiles);
        k_beta_pairwise.wait_front(chunk_key_tiles);  // beta*k*exp(G-anchor)
        beta_k.pop_front(chunk_key_tiles);
    }
    beta.pop_front(Ct);
    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_q, centered_decay, q_pairwise, chunk_key_tiles);
    q_pairwise.wait_front(chunk_key_tiles);  // q*exp(G-anchor)
    normalized_q.pop_front(chunk_key_tiles);
    centered_decay.pop_front(chunk_key_tiles);

    // The scan-facing factors exp(G) = exp(G-anchor) * exp(anchor) reuse the anchored products.
    pack_reconfig_data_format(q_pairwise.get_id(), q_decay.get_id());
    multiply_by_row(q_pairwise, anchor_decay, q_decay, chunk_key_tiles);
    pack_reconfig_data_format(q_decay.get_id(), kd.get_id());
    multiply_by_row(k_beta_pairwise, anchor_decay, kd, chunk_key_tiles);
    pack_reconfig_data_format(kd.get_id(), k_beta_pairwise.get_id());
}

// exp(G_last) = exp(G_last / 2)^2, squared in FP32 from the anchor's rows, of which the transpose to dl reads row 0.
template <uint32_t Ct, uint32_t Kt, typename DFB0, typename DFB1>
KDA_PREP_HELPER void prepare_final_decay_rows(DFB0& anchor_decay, DFB1& final_decay_rows) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    square_tiles(anchor_decay, final_decay_rows, chunk_key_tiles);
    final_decay_rows.wait_front(chunk_key_tiles);
}

template <uint32_t Ct, uint32_t Kt, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void prepare_k_pairwise(DFB0& normalized_k, DFB1& centered_inverse_decay, DFB2& k_pairwise) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    elementwise_binary<ElementwiseBinaryOp::Multiply>(
        normalized_k, centered_inverse_decay, k_pairwise, chunk_key_tiles);
    k_pairwise.wait_front(chunk_key_tiles);  // k*exp(anchor-G)
    normalized_k.pop_front(chunk_key_tiles);
    centered_inverse_decay.pop_front(chunk_key_tiles);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6>
KDA_PREP_HELPER void prepare_pairwise_matrices(
    DFB0& k_beta_pairwise, DFB1& q_pairwise, DFB2& k_pairwise, DFB3& causal_mask, DFB4& akk, DFB5& intra, DFB6& aqk) {
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

template <uint32_t Ct, uint32_t Kt, typename DFB0, typename DFB1, typename DFB2, typename DFB3, typename DFB4>
KDA_PREP_HELPER void prepare_decay_outputs(
    DFB0& k_pairwise, DFB1& anchor_decay, DFB2& final_decay_rows, DFB3& k_dec_t, DFB4& final_decay) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    // dl [K,1] is the transpose of any replicated exp(G_last) row.
    transpose_tile_row_to_column(final_decay_rows, final_decay, Kt);
    final_decay_rows.pop_front(chunk_key_tiles);

    {
        auto& k_dec = final_decay_rows;

        // k_dec_t = (kr * exp(G_last))^T.
        pack_reconfig_data_format(final_decay.get_id(), k_dec.get_id());
        multiply_by_row(k_pairwise, anchor_decay, k_dec, chunk_key_tiles);
        k_dec.wait_front(chunk_key_tiles);
        k_pairwise.pop_front(chunk_key_tiles);
        anchor_decay.pop_front(chunk_key_tiles);
        pack_reconfig_data_format(k_dec.get_id(), k_dec_t.get_id());
        transpose_tile_row_to_column(k_dec, k_dec_t, Kt);
        k_dec.pop_front(chunk_key_tiles);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t max_items, uint32_t SCALE_BITS, uint32_t EPS_BITS>
TT_KERNEL void compute(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks) {
    DataflowBuffer control(dfb::chronology_compute);
    const uint32_t valid_chunks = kda_chronology::receive(control).chunk_count();
    static_assert(Ct == 1, "chunk KDA currently requires chunk_size=32");

    constexpr uint32_t chunk_matrix_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;

    // Reader-produced inputs and constants.
    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    // Beta logits are activated into beta once per work item.
    DataflowBuffer beta_logits(dfb::beta);
    DataflowBuffer beta(dfb::beta_activated);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer block_masks(dfb::block_masks);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer gate_tril(dfb::gate_tril);
    DataflowBuffer gate_ones(dfb::gate_ones);

    // Writer-consumed outputs.
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);

    // Semantic intermediates.
    DataflowBuffer centered_inverse_decay(dfb::centered_inverse_decay);
    DataflowBuffer anchor_decay(dfb::anchor_decay);
    DataflowBuffer normalized_q(dfb::normalized_q);
    DataflowBuffer normalized_k(dfb::normalized_k);
    DataflowBuffer tile_workspace_0(dfb::tile_workspace_0);
    DataflowBuffer tile_workspace_1(dfb::tile_workspace_1);
    DataflowBuffer tile_workspace_2(dfb::tile_workspace_2);

    DataflowBuffer akk(dfb::akk);

    // Reusable physical storage. Live values crossing helper boundaries are named below.
    DataflowBuffer workspace_0(dfb::workspace_0);
    DataflowBuffer workspace_1(dfb::workspace_1);
    DataflowBuffer workspace_2(dfb::workspace_2);
    DataflowBuffer workspace_3(dfb::workspace_3);

    compute_kernel_hw_startup(dfb::q, dfb::k, dfb::workspace_3);
    // The reader generates the constants after publishing the first work item, the inverse's last; those are
    // awaited just before the batched inverse.
    ones.wait_front(chunk_matrix_tiles);
    gate_tril.wait_front(chunk_matrix_tiles);
    gate_ones.wait_front(chunk_matrix_tiles);
    tril.wait_front(chunk_matrix_tiles);

    uint32_t inverse_items = 0;
    for (uint32_t work_item = 0; work_item < work_item_count; ++work_item) {
        if ((work_item_start + work_item) % num_chunks >= valid_chunks) {
            continue;
        }
        q.wait_front(chunk_key_tiles);
        k.wait_front(chunk_key_tiles);
        v.wait_front(chunk_value_tiles);
        g.wait_front(chunk_key_tiles);
        beta_logits.wait_front(Ct);
        activate_beta<Ct>(beta_logits, beta);
        beta.wait_front(Ct);

        normalize_l2_rows<Ct, Kt, true, dfb::workspace_3, dfb::tile_workspace_0>(
            q,
            normalized_q,
            EPS_BITS,
            SCALE_BITS,
            /*squared=*/workspace_3,
            /*inverse_norms=*/tile_workspace_0);

        normalize_l2_rows<Ct, Kt, false, dfb::workspace_3, dfb::tile_workspace_0>(
            k,
            normalized_k,
            EPS_BITS,
            SCALE_BITS,
            /*squared=*/workspace_3,
            /*inverse_norms=*/tile_workspace_0);

        // PACK state persists across helpers. Reconfigure only at actual destination-format transitions.
        pack_reconfig_data_format(normalized_k.get_id(), v_beta.get_id());
        prepare_v_beta<Ct, Vt>(v, beta, v_beta);
        pack_reconfig_data_format(v_beta.get_id(), workspace_0.get_id());

        auto& centered_decay = workspace_0;
        prepare_gate_factors<Ct, Kt>(
            g,
            gate_tril,
            gate_ones,
            centered_decay,
            centered_inverse_decay,
            anchor_decay,
            /*anchor_g=*/workspace_2);

        auto& k_beta_pairwise = workspace_1;
        auto& q_pairwise = workspace_2;
        prepare_scan_and_pairwise_inputs<Ct, Kt>(
            normalized_q, normalized_k, beta, anchor_decay, centered_decay, q_decay, kd, k_beta_pairwise, q_pairwise);

        auto& final_decay_rows = workspace_0;
        prepare_final_decay_rows<Ct, Kt>(anchor_decay, final_decay_rows);

        auto& k_pairwise = workspace_3;
        prepare_k_pairwise<Ct, Kt>(normalized_k, centered_inverse_decay, k_pairwise);

        prepare_pairwise_matrices<Ct, Kt>(k_beta_pairwise, q_pairwise, k_pairwise, tril, akk, intra, tile_workspace_0);

        pack_reconfig_data_format(intra.get_id(), final_decay.get_id());
        prepare_decay_outputs<Ct, Kt>(k_pairwise, anchor_decay, final_decay_rows, k_decay_transposed, final_decay);
        pack_reconfig_data_format(k_decay_transposed.get_id(), workspace_3.get_id());
        // v_beta, kd, q_decay, intra, k_dec_t and dl stay pushed for the writer; Akk waits for the batched inverse.
        ++inverse_items;
    }
    if (inverse_items != 0) {
        eye.wait_front(chunk_matrix_tiles);
        block_masks.wait_front(3);
        // The inverse buffers hold max_items tiles. A batch of fewer items would let a later transaction cross a
        // ring end, which unpack and pack do not wrap, so the batch is always max_items: realign tile_workspace_0,
        // which the per-item normalizations and Aqk advanced, and append zero Akk tiles (their inverses are I and
        // the writer discards them).
        constexpr uint32_t workspace_0_tiles_per_item = 2 * Ct + Ct * Ct;
        for (uint32_t tile = workspace_0_tiles_per_item * inverse_items % max_items; tile != 0 && tile < max_items;
             ++tile) {
            tile_workspace_0.reserve_back(1);
            tile_workspace_0.push_back(1);
            tile_workspace_0.wait_front(1);
            tile_workspace_0.pop_front(1);
        }
        if (inverse_items != max_items) {
            binary_items(
                ElementwiseBinaryOp::Subtract, eye, 0, true, eye, 0, true, akk, (max_items - inverse_items) * Ct * Ct);
        }
        invert_items(
            akk,
            tril,
            eye,
            block_masks,
            t_inv,
            /*matrix=*/tile_workspace_0,
            /*total=*/tile_workspace_1,
            /*product=*/tile_workspace_2,
            max_items);
    }
}
