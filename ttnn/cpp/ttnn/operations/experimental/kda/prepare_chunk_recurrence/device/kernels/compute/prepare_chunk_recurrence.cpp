// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
//
// Chunk-parallel KDA prep. For cumulative vector gate G [C,K], factor
// exp(G_i-G_j) inside each q/k dot as exp(G_i)*exp(-G_j):
//   qd=q*exp(G), kl=k*exp(G), kr=k*exp(-G)
//   Akk=strictly_lower((beta*kl)@kr^T), Aqk=tril(qd@kr^T)
//   kd=beta*kl, k_dec_t=(k*exp(G_last-G))^T, dl=expm1(G_last)=exp(G_last)-1 (complement-form decay).
// Every exponent is a gate-scaled mask product formed and exponentiated in DST (FP32): G-a = (tril-ones/2)@g,
// a-G, a = (ones/2)@g, G_last-G = strict_upper@g and G_last = ones@g. BF16 g times a BF16-exact mask entry is exact
// and DST accumulates in FP32, so no exponent passes through a TF32 source register. (The cumulative sum G itself was
// already exact; the TF32 cut came from consuming it, about one TF32 ulp of |G| per exponent, so it grew with the
// decay rate. k_dec_t's exponent was moved into DST first, tt_metal_tracker-g1b.4.18.)
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
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Low FP32 mantissa bits an FPU consumer discards, by truncation, at HiFi3/HiFi4: a multiply keeps 9 explicit bits of
// its SrcA operand and 10 (TF32) of its SrcB operand (identity matmuls on Blackhole matched truncation bit-exactly,
// tt_metal_tracker-kxn.9.16); unpacking to SrcA for a copy also cuts to TF32. Producers round to nearest at these
// widths, so the later cut is exact and unbiased instead of dropping magnitude at every pass (about -0.07% per SrcA and
// -0.035% per SrcB pass, compounding to -0.4% on intra and T_inv). At HiFi2 SrcB keeps only 6 bits and these widths no
// longer make the cut exact.
constexpr uint32_t round_a = 23 - 9;
constexpr uint32_t round_b = 23 - 10;

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

enum class ElementwiseBinaryOp { Add, Subtract, Multiply };

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Round FP32 DST values to nearest (ties away from zero) at the given number of dropped mantissa bits:
// add half of the dropped range to the sign-magnitude bit pattern, then clear it. A carry into the exponent is
// the correct rounding of an all-ones mantissa.
inline void calculate_round_mantissa(uint32_t half, uint32_t keep_mask) {
    const sfpi::vInt half_v = static_cast<int>(half);
    const sfpi::vInt keep_v = static_cast<int>(keep_mask);
    for (int d = 0; d < 8; ++d) {
        sfpi::vInt bits = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        bits = (bits + half_v) & keep_v;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// Round DST tiles [first, first + count) to nearest at `drop` discarded mantissa bits (0: no-op), so the consumer's
// truncation (see round_a / round_b) is exact.
inline void round_dst_tiles(uint32_t first, uint32_t count, uint32_t drop) {
    if (drop == 0) {
        return;
    }
    [[maybe_unused]] const uint32_t half = 1U << (drop - 1);
    [[maybe_unused]] const uint32_t keep_mask = ~((1U << drop) - 1U);
    binop_with_scalar_tile_init();
    for (uint32_t tile = first; tile < first + count; ++tile) {
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_round_mantissa, tile, VectorMode::RC, half, keep_mask));
    }
}

// Compute out[Mt,Nt] = A[Mt,Kt] @ (transpose_b ? B[Nt,Kt]^T : B[Kt,Nt]) in the largest rectangular
// subblocks that exactly divide the output and fit in destination registers.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr>
inline void matmul_blocks(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o, uint32_t round_drop = 0) {
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
            if (round_drop) {
                round_dst_tiles(0, subblock_rows * subblock_columns, round_drop);
                matmul_block_init(a_id, b_id, Tr, subblock_columns, subblock_rows, Kt);
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
inline void elementwise_binary(
    DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o, uint32_t n, uint32_t round_drop = 0) {
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
        if (round_drop) {
            round_dst_tiles(0, block_tiles, round_drop);
            if constexpr (Op == ElementwiseBinaryOp::Add) {
                add_init(a_id, b_id);
            } else if constexpr (Op == ElementwiseBinaryOp::Subtract) {
                sub_init(a_id, b_id);
            } else {
                mul_init(a_id, b_id);
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

// Multiply one selected source-tile pair and publish the result.
// Mask products pass the exact mask first: a multiply's first operand lands in SrcA (8 explicit mantissa bits),
// its second in SrcB (10 bits at HiFi3/4), so the FP32 data keeps the wider multiplier input.
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

inline void square_tiles(DataflowBuffer& in, DataflowBuffer& o, uint32_t n, uint32_t round_drop = 0) {
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
        if (round_drop) {
            round_dst_tiles(0, block_tiles, round_drop);
            square_tile_init();
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
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
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

enum class MaskProductFunction { Exp, Expm1 };

// o[Ct,Kt] = f(mask @ g) for Ct = 1, with the product kept in DST (FP32) through f, so the exponent never passes
// through a source register. Expm1 is the complement-form decay: dl = expm1(G_last) keeps the relative precision of
// the forgotten fraction 1 - exp(G_last), which BF16 exp(G_last) loses for long-memory channels; recurrent_chunk_scan
// applies S + (dl * S + update). Mode and Iterations restrict exp to a row vector for row-replicated products:
// VectorMode::R visits faces 0 and 1, and two of a face's eight SFPU iterations span its first rows, row 0 included.
// The rest of each output tile is then unspecified.
template <uint32_t Kt, MaskProductFunction Function, VectorMode Mode = VectorMode::RC, int Iterations = 8>
inline void exponential_of_mask_products(
    DataflowBuffer& mask, DataflowBuffer& g, DataflowBuffer& o, uint32_t round_drop = 0) {
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
        if constexpr (Function == MaskProductFunction::Exp) {
            exp_tile_init();
            for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                exp_tile<false, false, InputClamping::ClampToNegative, Iterations>(tile, Mode);
            }
        } else {
            expm1_tile_init<false>();
            for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                expm1_tile<false>(tile);
            }
        }
        round_dst_tiles(0, block_tiles, round_drop);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, o_id, block_start + tile);
        }
        tile_regs_release();
    }
    o.push_back(Kt);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void multiply_by_column(
    DataflowBuffer& a, DataflowBuffer& col, DataflowBuffer& o, uint32_t Mt, uint32_t Nt, uint32_t round_drop = 0) {
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
        if (round_drop) {
            round_dst_tiles(0, block_tiles, round_drop);
            mul_bcast_cols_init(a_id, col_id);
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

// out[n] = A[n] * row[n]  (broadcast row 0 of each `row` tile down its tile)
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
            mul_tiles_bcast_rows(a_id, row_id, block_start + tile, block_start + tile, tile);
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
    multiply_selected_tile(block_masks, 0, negative_strict_lower_akk, 0, matrix);  // N1
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

    multiply_selected_tile(block_masks, 1, negative_strict_lower_akk, 0, matrix);  // L2
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

    multiply_selected_tile(block_masks, 2, negative_strict_lower_akk, 0, product);  // L3
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
        round_dst_tiles(dst, 1, round_b);
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

    square_tiles(input, squared, matrix_tiles, round_b);

    reduce_squared_rows_to_inverse_norms<Scale, SquaredDfb, InverseNormsDfb>(
        RowTiles, ColumnTiles, eps_bits, scale_bits);

    inverse_norms.wait_front(RowTiles);
    multiply_by_column(input, inverse_norms, normalized, RowTiles, ColumnTiles, round_a);
    normalized.wait_front(matrix_tiles);
    inverse_norms.pop_front(RowTiles);
    input.pop_front(matrix_tiles);
}

// beta = sigmoid(logits) in FP32, from the BF16 logits the reader moved into column 0. The SFPU sigmoid is the
// accurate one ttnn.sigmoid runs; BF16 widens exactly into the FP32 destination.
template <uint32_t Ct>
inline void activate_beta(DataflowBuffer& logits, DataflowBuffer& beta) {
    beta.reserve_back(Ct);
    reconfig_data_format_srca(logits.get_id());
    pack_reconfig_data_format(beta.get_id());
    copy_init(logits.get_id());
    sigmoid_tile_init<false>();
    for (uint32_t tile = 0; tile < Ct; ++tile) {
        tile_regs_acquire();
        copy_tile(logits.get_id(), tile, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        if (round_b) {
            round_dst_tiles(0, 1, round_b);
            sigmoid_tile_init<false>();
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, beta.get_id(), tile);
        tile_regs_release();
    }
    beta.push_back(Ct);
    logits.pop_front(Ct);
}

template <uint32_t Ct, uint32_t Vt>
inline void prepare_v_beta(DataflowBuffer& v, DataflowBuffer& beta, DataflowBuffer& v_beta) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    multiply_by_column(v, beta, v_beta, Ct, Vt);
    v.pop_front(chunk_value_tiles);
}

// Anchor the separable pairwise factors at a = G_last/2 so neither exp(G-a) nor exp(a-G) spans the full chunk range.
// Their products are unchanged, while realistic KDA gates no longer overflow exp(-G). g stays at the front for the
// final decay.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_gate_factors(
    DataflowBuffer& g,
    DataflowBuffer& centered_mask,
    DataflowBuffer& negated_centered_mask,
    DataflowBuffer& half_sum_mask,
    DataflowBuffer& strict_upper_mask,
    DataflowBuffer& centered_decay,
    DataflowBuffer& centered_inverse_decay,
    DataflowBuffer& anchor_decay,
    DataflowBuffer& key_suffix_decay) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    exponential_of_mask_products<Kt, MaskProductFunction::Exp>(
        centered_mask, g, centered_decay, round_b);  // exp(G - a)
    centered_decay.wait_front(chunk_key_tiles);
    exponential_of_mask_products<Kt, MaskProductFunction::Exp>(
        negated_centered_mask, g, centered_inverse_decay, round_b);  // exp(a - G)
    centered_inverse_decay.wait_front(chunk_key_tiles);
    // exp(a) is row-replicated and the broadcasts read only row 0, so only the face rows holding it are computed.
    // That takes two SFPU iterations: one leaves part of row 0 unwritten, and kd and q_decay come out wrong.
    exponential_of_mask_products<Kt, MaskProductFunction::Exp, VectorMode::R, 2>(
        half_sum_mask, g, anchor_decay, round_b);
    anchor_decay.wait_front(chunk_key_tiles);
    exponential_of_mask_products<Kt, MaskProductFunction::Exp>(
        strict_upper_mask, g, key_suffix_decay, round_b);  // exp(G_last - G)
    key_suffix_decay.wait_front(chunk_key_tiles);
}

template <uint32_t Ct, uint32_t Kt>
inline void prepare_scan_and_pairwise_inputs(
    DataflowBuffer& normalized_q,
    DataflowBuffer& normalized_k,
    DataflowBuffer& beta,
    DataflowBuffer& anchor_decay,
    DataflowBuffer& centered_decay,
    DataflowBuffer& q_decay,
    DataflowBuffer& kd,
    DataflowBuffer& k_beta_pairwise,
    DataflowBuffer& q_pairwise) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    {
        DataflowBuffer& beta_k = q_pairwise;
        multiply_by_column(normalized_k, beta, beta_k, Ct, Kt, round_a);
        beta_k.wait_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(
            beta_k, centered_decay, k_beta_pairwise, chunk_key_tiles, round_a);
        k_beta_pairwise.wait_front(chunk_key_tiles);  // beta*k*exp(G-anchor)
        beta_k.pop_front(chunk_key_tiles);
    }
    beta.pop_front(Ct);
    elementwise_binary<ElementwiseBinaryOp::Multiply>(
        normalized_q, centered_decay, q_pairwise, chunk_key_tiles, round_a);
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

// expm1(G_last) rows, replicated; the transpose to dl reads row 0.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_final_decay_rows(
    DataflowBuffer& g, DataflowBuffer& sum_mask, DataflowBuffer& anchor_decay, DataflowBuffer& final_decay_rows) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    exponential_of_mask_products<Kt, MaskProductFunction::Expm1>(sum_mask, g, final_decay_rows);
    final_decay_rows.wait_front(chunk_key_tiles);
    g.pop_front(chunk_key_tiles);
    anchor_decay.pop_front(chunk_key_tiles);  // its last consumer
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

    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_k, key_suffix_decay, k_dec, chunk_key_tiles, round_b);
    k_dec.wait_front(chunk_key_tiles);
    key_suffix_decay.pop_front(chunk_key_tiles);
    elementwise_binary<ElementwiseBinaryOp::Multiply>(
        normalized_k, centered_inverse_decay, k_pairwise, chunk_key_tiles, round_a);
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
    matmul_blocks<Ct, Kt, Ct, true>(k_beta_pairwise, k_pairwise, akk, round_b);
    akk.wait_front(chunk_matrix_tiles);  // raw beta*k_i*k_j*exp(G_i-G_j)
    k_beta_pairwise.pop_front(chunk_key_tiles);

    {
        matmul_blocks<Ct, Kt, Ct, true>(q_pairwise, k_pairwise, aqk, round_b);
        aqk.wait_front(chunk_matrix_tiles);  // raw q_i*k_j*exp(G_i-G_j)
        q_pairwise.pop_front(chunk_key_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(causal_mask, aqk, intra, chunk_matrix_tiles);
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
        elementwise_binary<ElementwiseBinaryOp::Multiply>(causal_mask, akk, lower_akk, chunk_matrix_tiles, round_b);
        lower_akk.wait_front(chunk_matrix_tiles);  // lower(A), including diagonal
        akk.pop_front(chunk_matrix_tiles);
        elementwise_binary<ElementwiseBinaryOp::Multiply>(
            identity, lower_akk, diagonal_akk, chunk_matrix_tiles, round_b);
        diagonal_akk.wait_front(chunk_matrix_tiles);  // diag(A)
        elementwise_binary<ElementwiseBinaryOp::Subtract>(diagonal_akk, lower_akk, akk, chunk_matrix_tiles, round_b);
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

// PACK must enter in final_decay's format and leaves in k_dec_t's.
template <uint32_t Ct, uint32_t Kt>
inline void prepare_decay_outputs(
    DataflowBuffer& k_dec, DataflowBuffer& final_decay_rows, DataflowBuffer& k_dec_t, DataflowBuffer& final_decay) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;

    // dl [K,1] is the transpose of any replicated exp(G_last) row.
    transpose_tile_row_to_column(final_decay_rows, final_decay, Kt);
    final_decay_rows.pop_front(chunk_key_tiles);

    // k_dec_t = (k * exp(G_last - G))^T.
    pack_reconfig_data_format(final_decay.get_id(), k_dec_t.get_id());
    transpose_tile_row_to_column(k_dec, k_dec_t, Kt);
    k_dec.pop_front(chunk_key_tiles);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t SCALE_BITS, uint32_t EPS_BITS>
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
    DataflowBuffer key_suffix_decay(dfb::key_suffix_decay);
    DataflowBuffer gate_strict_upper(dfb::gate_strict_upper);
    DataflowBuffer gate_half_ones(dfb::gate_half_ones);
    DataflowBuffer gate_centered(dfb::gate_centered);
    DataflowBuffer gate_centered_negated(dfb::gate_centered_negated);
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
    eye.wait_front(chunk_matrix_tiles);
    tril.wait_front(chunk_matrix_tiles);
    block_masks.wait_front(2);
    ones.wait_front(chunk_matrix_tiles);
    gate_tril.wait_front(chunk_matrix_tiles);
    gate_ones.wait_front(chunk_matrix_tiles);
    // Gate-scaled exponent masks, held for the whole kernel: strict_upper = ones - tril (G_last - G), half_ones
    // (a = G_last/2), centered = tril - half_ones (G - a) and its negation. With a BF16-exact gate scale s every entry
    // (s, s/2, -s/2, 0) is BF16-exact, so these TF32 operations are exact. PACK is in the FP32 workspace format from
    // the startup.
    elementwise_binary<ElementwiseBinaryOp::Subtract>(gate_ones, gate_tril, gate_strict_upper, chunk_matrix_tiles);
    gate_strict_upper.wait_front(chunk_matrix_tiles);
    multiply_by_half(gate_ones, gate_half_ones, chunk_matrix_tiles);
    gate_half_ones.wait_front(chunk_matrix_tiles);
    elementwise_binary<ElementwiseBinaryOp::Subtract>(gate_tril, gate_half_ones, gate_centered, chunk_matrix_tiles);
    gate_centered.wait_front(chunk_matrix_tiles);
    elementwise_binary<ElementwiseBinaryOp::Subtract>(
        gate_half_ones, gate_tril, gate_centered_negated, chunk_matrix_tiles);
    gate_centered_negated.wait_front(chunk_matrix_tiles);

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

        DataflowBuffer& centered_decay = workspace_0;
        prepare_gate_factors<Ct, Kt>(
            g,
            gate_centered,
            gate_centered_negated,
            gate_half_ones,
            gate_strict_upper,
            centered_decay,
            centered_inverse_decay,
            anchor_decay,
            key_suffix_decay);

        DataflowBuffer& k_beta_pairwise = workspace_1;
        DataflowBuffer& q_pairwise = workspace_2;
        prepare_scan_and_pairwise_inputs<Ct, Kt>(
            normalized_q, normalized_k, beta, anchor_decay, centered_decay, q_decay, kd, k_beta_pairwise, q_pairwise);

        DataflowBuffer& final_decay_rows = workspace_0;
        prepare_final_decay_rows<Ct, Kt>(g, gate_ones, anchor_decay, final_decay_rows);

        // anchor_decay's storage is free once final_decay_rows has consumed it; k_dec holds there until k_dec_t.
        DataflowBuffer& k_dec = anchor_decay;
        DataflowBuffer& k_pairwise = workspace_3;
        prepare_key_decays<Ct, Kt>(normalized_k, key_suffix_decay, centered_inverse_decay, k_dec, k_pairwise);

        prepare_pairwise_matrices<Ct, Kt>(k_beta_pairwise, q_pairwise, k_pairwise, tril, akk, intra, tile_workspace_0);
        k_pairwise.pop_front(chunk_key_tiles);

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

        pack_reconfig_data_format(t_inv.get_id(), final_decay.get_id());
        prepare_decay_outputs<Ct, Kt>(k_dec, final_decay_rows, k_decay_transposed, final_decay);
        pack_reconfig_data_format(k_decay_transposed.get_id(), workspace_3.get_id());
        // v_beta, kd, q_decay, intra, k_dec_t, dl, T_inv stay pushed for the writer.
    }
}
