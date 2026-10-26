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
#include "api/compute/copy_dest_values.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/engine_state.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/work_batches.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Code size: the whole program must fit the kernel config buffer, so each TRISC build keeps one copy of the code it
// repeats most. The macros only move code in or out of line; each tile sees the same operations.
//  - KDA_PREP_HELPER (the stage helpers): out of line in pack, whose loops stay rolled; inline in unpack and math.
//  - KDA_ITEMS_HELPER (helpers called from several stages): out of line everywhere; one copy in math and pack
//    (noclone), a copy per buffer-id specialization in unpack.
//  - KDA_MATH_OUTLINE: one copy of the matmul call and its init in math.
//  - KDA_PACK_OUTLINE: one copy of the pack loop in pack.
#ifdef TRISC_PACK
#define KDA_PREP_HELPER __attribute__((noinline))
#else
#define KDA_PREP_HELPER inline
#endif

#ifdef TRISC_UNPACK
#define KDA_ITEMS_HELPER __attribute__((noinline))
#else
#define KDA_ITEMS_HELPER __attribute__((noinline, noclone))
#endif

#ifdef TRISC_MATH
#define KDA_MATH_OUTLINE __attribute__((noinline, noclone))
#else
#define KDA_MATH_OUTLINE inline
#endif

#ifdef TRISC_PACK
#define KDA_PACK_OUTLINE __attribute__((noinline, noclone))
#else
#define KDA_PACK_OUTLINE inline
#endif

// Pack destination tiles [dst, dst + count) to o's tiles [out, out + count), in order.
KDA_PACK_OUTLINE void pack_tiles(uint32_t dst, uint32_t o_id, uint32_t out, uint32_t count) {
#pragma GCC unroll 1
    for (uint32_t tile = 0; tile < count; ++tile) {
        pack_tile(dst + tile, o_id, out + tile);
    }
}

KDA_MATH_OUTLINE void matmul_tiles_block(
    uint32_t a_id,
    uint32_t b_id,
    uint32_t a_tile,
    uint32_t b_tile,
    uint32_t dst,
    bool transpose,
    uint32_t ct,
    uint32_t rt,
    uint32_t kt) {
    matmul_block(a_id, b_id, a_tile, b_tile, dst, transpose, ct, rt, kt);
}

KDA_MATH_OUTLINE void matmul_tiles_block_init(
    uint32_t a_id, uint32_t b_id, bool transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    matmul_block_init(a_id, b_id, transpose, ct, rt, kt);
}

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

enum class ElementwiseBinaryOp { Add, Subtract, Multiply };

// The sources as compute_kernel_hw_startup leaves them (q, k). A stage reconfigures, and so drains the unpacker, only
// when its operands' formats differ; the configuration is the same either way. Global, so the helpers' signatures
// stay unchanged.
static kda::EngineState engine_state(dfb::q, dfb::k);

// Out of line, to keep the kernel small.
__attribute__((noinline)) void reconfig_tracked(uint32_t srca_operand, uint32_t srcb_operand) {
    engine_state.sources(srca_operand, srcb_operand);
}

// The unary stages reconfigure srcA unconditionally (see EngineState::reset_srca).
__attribute__((noinline)) void reconfig_tracked_srca(uint32_t srca_operand) { engine_state.reset_srca(srca_operand); }

// Every helper streams: each destination acquire waits only for the tiles it reads and publishes the tiles it packed,
// so consecutive stages overlap.

// Compute out_i[Mt,Nt] = A_i[Mt,Kt] @ (transpose_b ? B_i[Nt,Kt]^T : B_i[Kt,Nt]) for `items` independent products, in
// the largest rectangular subblocks that exactly divide the output and fit in destination registers. Item i's
// matrices follow item i-1's in each buffer; a shared A is the one matrix at the front of a.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void matmul_blocks(DFB0& a, bool a_shared, DFB1& b, DFB2& o, uint32_t items) {
    constexpr uint32_t subblock_columns = kda::MatmulSubblock<Mt, Nt>::columns;
    constexpr uint32_t subblock_rows = kda::MatmulSubblock<Mt, Nt>::rows;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(items * Mt * Nt);
    // Matmul maps in0=a_id->srcB and in1=b_id->srcA. Reconfigure unpack formats explicitly because init only
    // asserts formats; otherwise mixed fp32/bf16 CBs are read in the wrong format.
    reconfig_tracked(b_id, a_id);
    matmul_tiles_block_init(a_id, b_id, Tr, subblock_columns, subblock_rows, Kt);
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t a_base = a_shared ? 0 : item * Mt * Kt;
        const uint32_t b_base = item * Kt * Nt;
        const uint32_t o_base = item * Mt * Nt;
        a.wait_front(a_base + Mt * Kt);
        b.wait_front(b_base + Kt * Nt);
        for (uint32_t mi = 0; mi < Mt; mi += subblock_rows) {
            for (uint32_t ni = 0; ni < Nt; ni += subblock_columns) {
                tile_regs_acquire();
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    // kt_dim describes operand geometry; each K slice is still issued explicitly. B^T is stored
                    // [Nt,Kt], while the non-transposed B is stored [Kt,Nt].
                    const uint32_t b_index = Tr ? ni * Kt + ki : ki * Nt + ni;
                    matmul_tiles_block(
                        a_id,
                        b_id,
                        a_base + mi * Kt + ki,
                        b_base + b_index,
                        0,
                        Tr,
                        subblock_columns,
                        subblock_rows,
                        Kt);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t row = 0; row < subblock_rows; row++) {
                    pack_tiles(row * subblock_columns, o_id, o_base + (mi + row) * Nt + ni, subblock_columns);
                }
                tile_regs_release();
                o.push_back(subblock_rows * subblock_columns);
            }
        }
    }
}

// Tilewise binary operation, shared by every elementwise stage and by the batched inverse, where the core's work
// items run each stage together, up to `block` items per destination-register acquire, so the inverse pays
// its ~17 dependent stages once per core rather than once per item.
// a's tile for item i is a_base + i (or a_base alone when a is a shared constant), likewise b. Out of line, with the
// operation a runtime argument: the program sits at the kernel-config size limit.
template <typename DFB0, typename DFB1, typename DFB2>
KDA_ITEMS_HELPER void binary_items(
    ElementwiseBinaryOp op,
    DFB0& a,
    uint32_t a_base,
    bool a_shared,
    DFB1& b,
    uint32_t b_base,
    bool b_shared,
    DFB2& o,
    uint32_t n,
    uint32_t block) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(n);
    reconfig_tracked(a_id, b_id);
    if (op == ElementwiseBinaryOp::Add) {
        add_init(a_id, b_id);
    } else if (op == ElementwiseBinaryOp::Subtract) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t first = 0; first < n; first += block) {
        const uint32_t count = n - first < block ? n - first : block;
        if (!a_shared) {
            a.wait_front(a_base + first + count);
        }
        if (!b_shared) {
            b.wait_front(b_base + first + count);
        }
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
        pack_tiles(0, o_id, first, count);
        tile_regs_release();
        o.push_back(count);
    }
}

// Apply a binary operation tilewise to n tiles of a and b.
template <ElementwiseBinaryOp Op, typename DFB0, typename DFB1, typename DFB2>
inline void elementwise_binary(DFB0& a, DFB1& b, DFB2& o, uint32_t n) {
    binary_items(Op, a, 0, false, b, 0, false, o, n, max_dst_tiles);
}

// Mode and Iterations restrict the square to the first rows, as for the anchor's exponential (gate_exponentials).
template <VectorMode Mode = VectorMode::RC, int Iterations = 8, typename DFB0, typename DFB1>
KDA_PREP_HELPER void square_tiles(DFB0& in, DFB1& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_tracked_srca(in_id);
    copy_init(in_id);
    square_tile_init();
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        in.wait_front(block_start + block_tiles);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(in_id, block_start + tile, tile);
            square_tile<DST_ACCUM_MODE, Mode, Iterations>(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, block_start, block_tiles);
        tile_regs_release();
        o.push_back(block_tiles);
    }
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
template <typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void multiply_by_column(DFB0& a, DFB1& col, DFB2& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t col_id = col.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
    reconfig_tracked(a_id, col_id);  // bcast(a_id,col_id): a_id->srcA, col_id->srcB
    mul_bcast_cols_init(a_id, col_id);
    const uint32_t output_tiles = Mt * Nt;
    for (uint32_t block_start = 0; block_start < output_tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles =
            (output_tiles - block_start < max_dst_tiles) ? output_tiles - block_start : max_dst_tiles;
        a.wait_front(block_start + block_tiles);
        col.wait_front((block_start + block_tiles - 1) / Nt + 1);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t input_tile = block_start + tile;
            mul_tiles_bcast_cols(a_id, col_id, input_tile, input_tile / Nt, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, block_start, block_tiles);
        tile_regs_release();
        o.push_back(block_tiles);
    }
}

// out[Mt,Nt] = (a[Mt,Nt] * col[Mt,1], broadcast across N) * b[Mt,Nt]: the column product stays in DST and the second
// product takes it as srcA (in FP32, b's format).
template <typename DFB0, typename DFB1, typename DFB2, typename DFB3>
KDA_PREP_HELPER void multiply_by_column_then(DFB0& a, DFB1& col, DFB2& b, DFB3& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t col_id = col.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();
    const uint32_t output_tiles = Mt * Nt;
    o.reserve_back(output_tiles);
    for (uint32_t block_start = 0; block_start < output_tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles =
            (output_tiles - block_start < max_dst_tiles) ? output_tiles - block_start : max_dst_tiles;
        a.wait_front(block_start + block_tiles);
        col.wait_front((block_start + block_tiles - 1) / Nt + 1);
        b.wait_front(block_start + block_tiles);
        tile_regs_acquire();
        reconfig_tracked(a_id, col_id);
        mul_bcast_cols_init(a_id, col_id);
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t input_tile = block_start + tile;
            mul_tiles_bcast_cols(a_id, col_id, input_tile, input_tile / Nt, tile);
        }
        reconfig_tracked(b_id, b_id);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(b_id);
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(b_id, block_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, block_start, block_tiles);
        tile_regs_release();
        o.push_back(block_tiles);
    }
}

// out_i = (a_i[1,Kt] @ b_i[1,Kt]^T) * mask, one tile per item: each product stays in DST and the mask multiplies it
// as srcA (FP32). A destination acquire holds up to max_dst_tiles items' products.
template <uint32_t Kt, typename DFB0, typename DFB1, typename DFB2, typename DFB3>
KDA_PREP_HELPER void matmul_transposed_masked(DFB0& a, DFB1& b, DFB2& mask, DFB3& o, uint32_t items) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t mask_id = mask.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(items);
    for (uint32_t first = 0; first < items; first += max_dst_tiles) {
        const uint32_t count = items - first < max_dst_tiles ? items - first : max_dst_tiles;
        a.wait_front((first + count) * Kt);
        b.wait_front((first + count) * Kt);
        reconfig_tracked(b_id, a_id);
        matmul_tiles_block_init(a_id, b_id, true, 1, 1, Kt);
        tile_regs_acquire();
        for (uint32_t item = 0; item < count; ++item) {
            const uint32_t base = (first + item) * Kt;
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_tiles_block(a_id, b_id, base + k, base + k, item, true, 1, 1, Kt);
            }
        }
        reconfig_tracked(mask_id, mask_id);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(mask_id);
        for (uint32_t item = 0; item < count; ++item) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(mask_id, 0, item);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, first, count);
        tile_regs_release();
        o.push_back(count);
    }
}

// out[n] = A[n] * row[n]  (broadcast row 0 of each `row` tile down its tile)
template <typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void multiply_by_row(DFB0& a, DFB1& row, DFB2& o, uint32_t n) {
    const uint32_t a_id = a.get_id();
    const uint32_t row_id = row.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    reconfig_tracked(a_id, row_id);  // bcast(a_id,row_id): a_id->srcA, row_id->srcB
    mul_bcast_rows_init(a_id, row_id);
    for (uint32_t block_start = 0; block_start < n; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (n - block_start < max_dst_tiles) ? n - block_start : max_dst_tiles;
        a.wait_front(block_start + block_tiles);
        row.wait_front(block_start + block_tiles);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            mul_tiles_bcast_rows(a_id, row_id, block_start + tile, block_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, block_start, block_tiles);
        tile_regs_release();
        o.push_back(block_tiles);
    }
}

// o[i] = a[i] @ b[i] for one-tile matrices, up to `block` items per destination-register acquire.
template <typename DFB0, typename DFB1, typename DFB2>
KDA_ITEMS_HELPER void matmul_items(DFB0& a, DFB1& b, DFB2& o, uint32_t n, uint32_t block) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(n);
    reconfig_tracked(b_id, a_id);
    matmul_tiles_block_init(a_id, b_id, false, 1, 1, 1);
    for (uint32_t first = 0; first < n; first += block) {
        const uint32_t count = n - first < block ? n - first : block;
        a.wait_front(first + count);
        b.wait_front(first + count);
        tile_regs_acquire();
        for (uint32_t item = 0; item < count; ++item) {
            matmul_tiles_block(a_id, b_id, first + item, first + item, item, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, first, count);
        tile_regs_release();
        o.push_back(count);
    }
}

// T_inv = (I + strictly_lower(Akk))^-1 for the n items whose raw Akk tiles are the front of `akk`.
// Invert (I-N) for the strictly lower N = -strictly_lower(Akk) by nesting three block levels, so every matmul keeps
// one strictly-lower factor and no large cancelling power of PS4 is ever formed:
//   level 1: 4-row diagonal blocks. N1^4=0, so A = (I-N1)^-1 = I+N1(I+N1(I+N1)) in two matmuls.
//   level 2: pair those blocks. M2 = A L2 satisfies M2^2=0, so B = (I+M2)A = A + M2 A.
//   level 3: join the 8-row blocks. M3 = B L3 satisfies M3^4=0, so (I-N)^-1 = (I+M3)(I+M3^2)B.
// Each block mask holds -1 on its level's strictly lower entries, so N1, L2 and L3 are each one product with the raw
// Akk; a product with 0 or -1 is exact.
// The items are independent: each stage runs over all n of them, its tile for item i at the front of its buffers + i.
template <typename DFB0, typename DFB1, typename DFB2, typename DFB3, typename DFB4, typename DFB5, typename DFB6>
__attribute__((noinline)) void invert_items(
    DFB0& akk, DFB1& identity, DFB2& block_masks, DFB3& inverse, DFB4& matrix, DFB5& total, DFB6& product, uint32_t n) {
    using Op = ElementwiseBinaryOp;
    // Two-tile acquires: each dependent stage starts on the previous one's first tiles while it packs the rest.
    constexpr uint32_t inverse_block = 2;
    binary_items(Op::Multiply, akk, 0, false, block_masks, 0, true, matrix, n, inverse_block);  // N1
    binary_items(Op::Add, identity, 0, true, matrix, 0, false, total, n, inverse_block);
    for (uint32_t step = 0; step < 2; ++step) {
        matmul_items(matrix, total, product, n, inverse_block);
        // The double-depth total holds the old and new Horner values together.
        binary_items(Op::Add, identity, 0, true, product, 0, false, total, n, inverse_block);
        total.pop_front(n);
        product.pop_front(n);
    }
    matrix.pop_front(n);  // total holds A

    binary_items(Op::Multiply, akk, 0, false, block_masks, 1, true, matrix, n, inverse_block);  // L2
    matmul_items(total, matrix, product, n, inverse_block);                                     // M2 = A L2
    matrix.pop_front(n);
    matmul_items(product, total, matrix, n, inverse_block);  // M2 A
    product.pop_front(n);
    binary_items(Op::Add, total, 0, false, matrix, 0, false, total, n, inverse_block);  // B = A + M2 A
    total.pop_front(n);
    matrix.pop_front(n);

    binary_items(Op::Multiply, akk, 0, false, block_masks, 2, true, product, n, inverse_block);  // L3
    // Akk is dead after extracting L3; its buffer holds I+M3 below.
    akk.pop_front(n);
    auto& identity_plus_m3 = akk;
    matmul_items(total, product, matrix, n, inverse_block);  // M3 = B L3
    product.pop_front(n);
    matmul_items(matrix, matrix, product, n, inverse_block);                                         // M3^2
    binary_items(Op::Add, identity, 0, true, matrix, 0, false, identity_plus_m3, n, inverse_block);  // I+M3
    matrix.pop_front(n);
    binary_items(Op::Add, identity, 0, true, product, 0, false, matrix, n, inverse_block);  // I+M3^2
    product.pop_front(n);
    matmul_items(identity_plus_m3, matrix, product, n, inverse_block);  // (I+M3)(I+M3^2)
    identity_plus_m3.pop_front(n);
    matrix.pop_front(n);
    matmul_items(product, total, inverse, n, inverse_block);
    product.pop_front(n);
    total.pop_front(n);
}

// Transpose a tiled row [1,row_tiles] into a tiled column [row_tiles,1].
template <typename DFB0, typename DFB1>
KDA_PREP_HELPER void transpose_tile_row_to_column(DFB0& in, DFB1& o, uint32_t row_tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(row_tiles);
    reconfig_tracked_srca(in_id);  // unary: in_id->srcA
    transpose_init(in_id);
    for (uint32_t block_start = 0; block_start < row_tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles =
            (row_tiles - block_start < max_dst_tiles) ? row_tiles - block_start : max_dst_tiles;
        in.wait_front(block_start + block_tiles);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            transpose_tile(in_id, block_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, block_start, block_tiles);
        tile_regs_release();
        o.push_back(block_tiles);
    }
}

// Reduce each row of squared values directly to its inverse-L2 factor. Applying epsilon, rsqrt,
// and optional scaling in DST avoids materializing row sums in a separate intermediate DFB.
// The row sums, and the column broadcast that applies the factors, use only column 0, so the epilogue runs on the
// faces holding it (VectorMode::C); the other columns of each factor tile are unspecified.
template <uint32_t SquaredDfb, uint32_t InverseNormsDfb>
KDA_ITEMS_HELPER void reduce_squared_rows_to_inverse_norms(
    uint32_t row_tiles, uint32_t column_tiles, uint32_t eps_bits, bool scale, uint32_t scale_bits) {
    auto inverse_norm = [eps_bits, scale, scale_bits](uint32_t dst) {
        binop_with_scalar_tile_init();
        add_unary_tile(dst, eps_bits, VectorMode::C);
        rsqrt_tile_init();
        rsqrt_tile(dst, VectorMode::C);
        if (scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(dst, scale_bits, VectorMode::C);
        }
    };

    // The squares and the ones share the FP32 format, and the packer is already in it.
    reconfig_tracked(SquaredDfb, dfb::ones);
    compute_kernel_lib::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        SquaredDfb,
        dfb::ones,
        InverseNormsDfb,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
        compute_kernel_lib::ReduceInputBlockShape::of(row_tiles, column_tiles),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        inverse_norm);
}

template <
    uint32_t ColumnTiles,
    uint32_t SquaredDfb,
    uint32_t InverseNormsDfb,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3>
KDA_PREP_HELPER void normalize_l2_rows(
    DFB0& input,
    DFB1& normalized,
    uint32_t row_tiles,
    uint32_t eps_bits,
    bool scale,
    uint32_t scale_bits,

    // intermediate
    DFB2& squared,
    DFB3& inverse_norms) {
    const uint32_t matrix_tiles = row_tiles * ColumnTiles;

    // The squares on the FPU: the HiFi4 product of a BF16 value with itself is exact in FP32, as the SFPU square was.
    binary_items(ElementwiseBinaryOp::Multiply, input, 0, false, input, 0, false, squared, matrix_tiles, max_dst_tiles);

    reduce_squared_rows_to_inverse_norms<SquaredDfb, InverseNormsDfb>(
        row_tiles, ColumnTiles, eps_bits, scale, scale_bits);

    multiply_by_column(input, inverse_norms, normalized, row_tiles, ColumnTiles);
    inverse_norms.pop_front(row_tiles);
    input.pop_front(matrix_tiles);
}

// beta = sigmoid(logits) in FP32, from the BF16 logits the reader moved into row 0, transposed into the column 0
// the broadcasts read. The SFPU sigmoid is the accurate one ttnn.sigmoid runs, on the face rows holding row 0 only
// (two SFPU iterations, as for the anchor's exponential); BF16 widens exactly into the FP32 destination, and the FP32
// transpose is exact. The rest of each beta tile is unspecified.
template <uint32_t Ct, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void activate_beta(DFB0& logits, DFB1& beta_row, DFB2& beta, uint32_t items) {
    const uint32_t tiles = items * Ct;
    const uint32_t logits_id = logits.get_id();
    const uint32_t row_id = beta_row.get_id();
    beta_row.reserve_back(tiles);
    reconfig_tracked_srca(logits_id);
    copy_init(logits_id);
    sigmoid_tile_init<false>();
    for (uint32_t block_start = 0; block_start < tiles; block_start += max_dst_tiles) {
        const uint32_t block_tiles = (tiles - block_start < max_dst_tiles) ? tiles - block_start : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(logits_id, block_start + tile, tile);
            sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, 2>(tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, row_id, block_start, block_tiles);
        tile_regs_release();
        beta_row.push_back(block_tiles);
    }
    logits.pop_front(tiles);
    transpose_tile_row_to_column(beta_row, beta, tiles);
    beta_row.pop_front(tiles);
}

template <uint32_t Ct, uint32_t Vt, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void prepare_v_beta(DFB0& v, DFB1& beta, DFB2& v_beta, uint32_t items) {
    multiply_by_column(v, beta, v_beta, items * Ct, Vt);
    v.pop_front(items * Ct * Vt);
}

// out_i = a @ b_i - c_i for a shared one-tile a and each item's row of Nt tiles b_i: the product stays in DST and the
// subtraction takes it as srcA (in FP32, c's format), so it never round-trips through a buffer.
template <uint32_t Nt, typename DFB0, typename DFB1, typename DFB2, typename DFB3>
KDA_PREP_HELPER void matmul_row_minus(DFB0& a, DFB1& b, DFB2& c, DFB3& o, uint32_t items) {
    static_assert(Nt <= max_dst_tiles, "the product row fills one DST acquire");
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t c_id = c.get_id();
    const uint32_t o_id = o.get_id();
    o.reserve_back(items * Nt);
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t base = item * Nt;
        c.wait_front(base + Nt);
        reconfig_tracked(b_id, a_id);
        matmul_tiles_block_init(a_id, b_id, false, Nt, 1, 1);
        tile_regs_acquire();
        matmul_tiles_block(a_id, b_id, 0, base, 0, false, Nt, 1, 1);
        reconfig_tracked(c_id, c_id);
        sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_id);
        for (uint32_t tile = 0; tile < Nt; ++tile) {
            sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_id, base + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, o_id, base, Nt);
        tile_regs_release();
        o.push_back(Nt);
    }
}

// exp(x) into exponential and its precise reciprocal exp(-x) into inverse, from one unpack of x (both FP32-accurate),
// and the anchor's row exponential into anchor_exponential. The anchor is row-replicated and the broadcasts read only
// row 0, so only the face rows holding it are computed: VectorMode::R visits faces 0 and 1, and two of a face's eight
// SFPU iterations span its first rows, row 0 included (one leaves part of row 0 unwritten, and kd, q_decay, k_dec_t
// and final_decay come out wrong); the rest of each anchor_exponential tile is unspecified.
// One tile of each per destination acquire: the anchor's unpack and pack then overlap the exponential and reciprocal
// of the neighbouring acquires instead of running as a separate stage.
template <typename DFB0, typename DFB1, typename DFB2, typename DFB3, typename DFB4>
KDA_PREP_HELPER void gate_exponentials(
    DFB0& in, DFB1& anchor, DFB2& exponential, DFB3& inverse, DFB4& anchor_exponential, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t anchor_id = anchor.get_id();
    const uint32_t exp_id = exponential.get_id();
    const uint32_t inv_id = inverse.get_id();
    const uint32_t anchor_exp_id = anchor_exponential.get_id();
    constexpr auto recip_dest = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16;
    exponential.reserve_back(n);
    inverse.reserve_back(n);
    anchor_exponential.reserve_back(n);
    reconfig_tracked_srca(in_id);
    copy_init(in_id);
    for (uint32_t tile = 0; tile < n; ++tile) {
        in.wait_front(tile + 1);
        anchor.wait_front(tile + 1);
        tile_regs_acquire();
        exp_tile_init();
        copy_tile(in_id, tile, 0);
        exp_tile<false, false, InputClamping::ClampToNegative>(0, VectorMode::RC);
        copy_tile(anchor_id, tile, 2);
        exp_tile<false, false, InputClamping::ClampToNegative, 2>(2, VectorMode::R);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        recip_tile_init<recip_dest, ReciprocalApproxMode::Precise>();
        recip_tile<recip_dest, ReciprocalApproxMode::Precise>(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tiles(0, exp_id, tile, 1);
        pack_tiles(1, inv_id, tile, 1);
        pack_tiles(2, anchor_exp_id, tile, 1);
        tile_regs_release();
        exponential.push_back(1);
        inverse.push_back(1);
        anchor_exponential.push_back(1);
    }
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
    typename DFB7>
KDA_PREP_HELPER void prepare_gate_factors(
    DFB0& g,
    DFB1& prefix_sum_mask,
    DFB2& sum_broadcast_matrix,
    DFB3& centered_decay,
    DFB4& centered_inverse_decay,
    DFB5& anchor_decay,
    uint32_t items,

    // intermediate
    DFB6& centered_g,
    DFB7& anchor_g) {
    const uint32_t key_tiles = items * Ct * Kt;

    // G = cumsum(g). Anchor the separable pairwise factors at G_last/2 so neither
    // exp(G-anchor) nor exp(anchor-G) spans the full chunk range: their products are
    // those of exp(G) and exp(-G), which realistic KDA gates would overflow.
    static_assert(Ct == 1, "the gate factors take one tile row per chunk");
    // The sum-broadcast matrix carries half the gate scale, so this is the anchor G_last/2, replicated.
    matmul_blocks<Ct, Ct, Kt, false>(sum_broadcast_matrix, /*a_shared=*/true, g, anchor_g, items);

    // G - anchor, the prefix sum kept in DST.
    matmul_row_minus<Kt>(prefix_sum_mask, g, anchor_g, centered_g, items);
    g.pop_front(key_tiles);

    // exp(G-anchor), its precise reciprocal exp(anchor-G) and the row exponential exp(G_last/2). The anchor has its own
    // buffer, so centered_decay can be packed while the anchor is still read.
    gate_exponentials(centered_g, anchor_g, centered_decay, centered_inverse_decay, anchor_decay, key_tiles);
    centered_g.pop_front(key_tiles);
    anchor_g.pop_front(key_tiles);
}

// dl [K,1] = exp(G_last) as a column: exp(G_last / 2)^2, squared in FP32 from the anchor's rows, then the transpose of
// row 0. Only the face rows holding row 0 are squared (the anchor's exponential computed only those); the rest of
// each rows tile, and so every column of dl but column 0, is unspecified. Leaves the packer in the rows' format.
template <uint32_t Ct, uint32_t Kt, typename DFB0, typename DFB1, typename DFB2>
KDA_PREP_HELPER void prepare_final_decay(
    DFB0& anchor_decay,
    DFB1& final_decay,
    uint32_t items,

    // intermediate
    DFB2& final_decay_rows) {
    const uint32_t key_tiles = items * Ct * Kt;
    square_tiles<VectorMode::R, 2>(anchor_decay, final_decay_rows, key_tiles);
    pack_reconfig_data_format(final_decay_rows.get_id(), final_decay.get_id());
    transpose_tile_row_to_column(final_decay_rows, final_decay, items * Kt);
    final_decay_rows.pop_front(key_tiles);
    pack_reconfig_data_format(final_decay.get_id(), final_decay_rows.get_id());
}

// q_decay = q*exp(G) from the normalized q, through q_pairwise = q*exp(G-anchor), which the pairwise product reads.
// The packer starts and ends in the FP32 intermediates' format.
template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t SquaredDfb,
    uint32_t InverseNormsDfb,
    uint32_t SCALE_BITS,
    uint32_t EPS_BITS,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6,
    typename DFB7>
KDA_PREP_HELPER void prepare_q_terms(
    DFB0& q,
    DFB1& centered_decay,
    DFB2& anchor_decay,
    DFB3& q_decay,
    DFB4& q_pairwise,
    uint32_t items,

    // intermediate
    DFB5& normalized_q,
    DFB6& squared,
    DFB7& inverse_norms) {
    const uint32_t key_tiles = items * Ct * Kt;

    normalize_l2_rows<Kt, SquaredDfb, InverseNormsDfb>(
        q, normalized_q, items * Ct, EPS_BITS, /*scale=*/true, SCALE_BITS, squared, inverse_norms);
    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_q, centered_decay, q_pairwise, key_tiles);
    normalized_q.pop_front(key_tiles);  // q*exp(G-anchor) is packed

    // The scan-facing factor exp(G) = exp(G-anchor) * exp(anchor) reuses the anchored product.
    pack_reconfig_data_format(q_pairwise.get_id(), q_decay.get_id());
    multiply_by_row(q_pairwise, anchor_decay, q_decay, key_tiles);
    pack_reconfig_data_format(q_decay.get_id(), q_pairwise.get_id());
}

// kd = beta*k*exp(G) from the normalized k, through k_beta_pairwise = beta*k*exp(G-anchor) and
// k_pairwise = k*exp(anchor-G), which the pairwise products read. Releases beta and both centered decays. The packer
// starts and ends in the FP32 intermediates' format.
template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t SquaredDfb,
    uint32_t InverseNormsDfb,
    uint32_t EPS_BITS,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5,
    typename DFB6,
    typename DFB7,
    typename DFB8,
    typename DFB9,
    typename DFB10>
KDA_PREP_HELPER void prepare_k_terms(
    DFB0& k,
    DFB1& beta,
    DFB2& centered_decay,
    DFB3& centered_inverse_decay,
    DFB4& anchor_decay,
    DFB5& kd,
    DFB6& k_beta_pairwise,
    DFB7& k_pairwise,
    uint32_t items,

    // intermediate
    DFB8& normalized_k,
    DFB9& squared,
    DFB10& inverse_norms) {
    const uint32_t key_tiles = items * Ct * Kt;

    normalize_l2_rows<Kt, SquaredDfb, InverseNormsDfb>(
        k, normalized_k, items * Ct, EPS_BITS, /*scale=*/false, /*scale_bits=*/0, squared, inverse_norms);

    // beta*k*exp(G-anchor), beta*k kept in DST.
    multiply_by_column_then(normalized_k, beta, centered_decay, k_beta_pairwise, items * Ct, Kt);
    beta.pop_front(items * Ct);
    centered_decay.pop_front(key_tiles);

    elementwise_binary<ElementwiseBinaryOp::Multiply>(normalized_k, centered_inverse_decay, k_pairwise, key_tiles);
    normalized_k.pop_front(key_tiles);  // k*exp(anchor-G) is packed
    centered_inverse_decay.pop_front(key_tiles);

    pack_reconfig_data_format(k_pairwise.get_id(), kd.get_id());
    multiply_by_row(k_beta_pairwise, anchor_decay, kd, key_tiles);
    pack_reconfig_data_format(kd.get_id(), k_beta_pairwise.get_id());
}

template <
    uint32_t Ct,
    uint32_t Kt,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3,
    typename DFB4,
    typename DFB5>
KDA_PREP_HELPER void prepare_pairwise_matrices(
    DFB0& k_beta_pairwise,
    DFB1& q_pairwise,
    DFB2& k_pairwise,
    DFB3& causal_mask,
    DFB4& akk,
    DFB5& intra,
    uint32_t items) {
    const uint32_t key_tiles = items * Ct * Kt;

    // Raw beta*k_i*k_j*exp(G_i-G_j), which the batched inverse reads after the last batch.
    matmul_blocks<Ct, Kt, Ct, true>(k_beta_pairwise, /*a_shared=*/false, k_pairwise, akk, items);
    k_beta_pairwise.pop_front(key_tiles);

    // tril(q_i*k_j*exp(G_i-G_j)), the raw product kept in DST.
    static_assert(Ct == 1, "the masked pairwise product is one tile per chunk");
    matmul_transposed_masked<Kt>(q_pairwise, k_pairwise, causal_mask, intra, items);
    q_pairwise.pop_front(key_tiles);
}

// k_dec_t = (kr * exp(G_last))^T, from k_pairwise and the anchor's rows, which it releases. Leaves the packer in the
// intermediate's format.
template <uint32_t Ct, uint32_t Kt, typename DFB0, typename DFB1, typename DFB2, typename DFB3>
KDA_PREP_HELPER void prepare_k_decay_transposed(
    DFB0& k_pairwise,
    DFB1& anchor_decay,
    DFB2& k_dec_t,
    uint32_t items,

    // intermediate
    DFB3& k_dec) {
    const uint32_t key_tiles = items * Ct * Kt;
    multiply_by_row(k_pairwise, anchor_decay, k_dec, key_tiles);
    k_pairwise.pop_front(key_tiles);
    anchor_decay.pop_front(key_tiles);
    pack_reconfig_data_format(k_dec.get_id(), k_dec_t.get_id());
    transpose_tile_row_to_column(k_dec, k_dec_t, items * Kt);
    k_dec.pop_front(key_tiles);
    pack_reconfig_data_format(k_dec_t.get_id(), k_dec.get_id());
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t max_items,
    uint32_t batch_items,
    uint32_t SCALE_BITS,
    uint32_t EPS_BITS>
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
    DataflowBuffer beta_row(dfb::beta_row);
    DataflowBuffer beta(dfb::beta_activated);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer block_masks(dfb::block_masks);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer gate_tril(dfb::gate_tril);
    DataflowBuffer gate_ones(dfb::gate_ones);

    // Outputs, drained by the writer (and v_beta, k_decay_transposed by the reader).
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
    DataflowBuffer normalized_k(dfb::normalized_k);
    DataflowBuffer inverse_norms(dfb::inverse_norms);
    DataflowBuffer tile_workspace_0(dfb::tile_workspace_0);
    DataflowBuffer tile_workspace_1(dfb::tile_workspace_1);
    DataflowBuffer tile_workspace_2(dfb::tile_workspace_2);

    DataflowBuffer akk(dfb::akk);

    // Reusable physical storage. Live values crossing helper boundaries are named below.
    DataflowBuffer workspace_0(dfb::workspace_0);
    DataflowBuffer workspace_1(dfb::workspace_1);
    DataflowBuffer workspace_2(dfb::workspace_2);

    compute_kernel_hw_startup(dfb::q, dfb::k, dfb::workspace_0);
    // The writer generates every constant at startup, in order of first use; each is awaited before its first use.
    gate_tril.wait_front(chunk_matrix_tiles);
    gate_ones.wait_front(chunk_matrix_tiles);

    const uint32_t inverse_items = kda_prep::count_valid(work_item_start, work_item_count, num_chunks, valid_chunks);
    // Stage by stage over batches of batch_items; the items past the last whole batch run one at a time, so every
    // reservation stays inside its one-batch ring (unpack and pack do not wrap). The PACK format persists across
    // helpers, which leave it in the FP32 intermediates' format.
    for (uint32_t done = 0; done < inverse_items;) {
        const uint32_t items = kda_prep::batch_size<batch_items>(inverse_items - done);
        const uint32_t key_tiles = items * chunk_key_tiles;

        auto& centered_decay = workspace_0;
        g.wait_front(key_tiles);
        prepare_gate_factors<Ct, Kt>(
            g,
            gate_tril,
            gate_ones,
            centered_decay,
            centered_inverse_decay,
            anchor_decay,
            items,
            /*centered_g=*/workspace_1,
            /*anchor_g=*/workspace_2);

        prepare_final_decay<Ct, Kt>(anchor_decay, final_decay, items, /*final_decay_rows=*/workspace_1);

        beta_logits.wait_front(items * Ct);
        activate_beta<Ct>(beta_logits, beta_row, beta, items);

        // q and k normalize through the same squared and normalized buffers, so one copy of the row reduction serves
        // both.
        auto& q_pairwise = workspace_1;
        ones.wait_front(chunk_matrix_tiles);
        tril.wait_front(chunk_matrix_tiles);
        q.wait_front(key_tiles);
        prepare_q_terms<Ct, Kt, dfb::workspace_2, dfb::inverse_norms, SCALE_BITS, EPS_BITS>(
            q,
            centered_decay,
            anchor_decay,
            q_decay,
            q_pairwise,
            items,
            /*normalized_q=*/normalized_k,
            /*squared=*/workspace_2,
            inverse_norms);

        v.wait_front(items * chunk_value_tiles);
        pack_reconfig_data_format(q_pairwise.get_id(), v_beta.get_id());
        prepare_v_beta<Ct, Vt>(v, beta, v_beta, items);
        pack_reconfig_data_format(v_beta.get_id(), workspace_2.get_id());

        auto& k_beta_pairwise = workspace_2;
        auto& k_pairwise = workspace_0;
        k.wait_front(key_tiles);
        prepare_k_terms<Ct, Kt, dfb::workspace_2, dfb::inverse_norms, EPS_BITS>(
            k,
            beta,
            centered_decay,
            centered_inverse_decay,
            anchor_decay,
            kd,
            k_beta_pairwise,
            k_pairwise,
            items,
            normalized_k,
            /*squared=*/workspace_2,
            inverse_norms);

        prepare_pairwise_matrices<Ct, Kt>(k_beta_pairwise, q_pairwise, k_pairwise, tril, akk, intra, items);

        pack_reconfig_data_format(intra.get_id(), workspace_1.get_id());
        prepare_k_decay_transposed<Ct, Kt>(k_pairwise, anchor_decay, k_decay_transposed, items, /*k_dec=*/workspace_1);
        // The per-batch outputs stay pushed for the dataflow kernels; Akk waits for the batched inverse.
        done += items;
    }
    if (inverse_items != 0) {
        eye.wait_front(chunk_matrix_tiles);
        block_masks.wait_front(3);
        // The inverse buffers hold max_items tiles. A batch of fewer items would let a later transaction cross a
        // ring end, which unpack and pack do not wrap, so the batch is always max_items: append zero Akk tiles
        // (their inverses are I and the writer discards them).
        if (inverse_items != max_items) {
            binary_items(
                ElementwiseBinaryOp::Subtract,
                eye,
                0,
                true,
                eye,
                0,
                true,
                akk,
                (max_items - inverse_items) * Ct * Ct,
                max_dst_tiles);
        }
        invert_items(
            akk,
            eye,
            block_masks,
            t_inv,
            /*matrix=*/tile_workspace_0,
            /*total=*/tile_workspace_1,
            /*product=*/tile_workspace_2,
            max_items);
    }
}
