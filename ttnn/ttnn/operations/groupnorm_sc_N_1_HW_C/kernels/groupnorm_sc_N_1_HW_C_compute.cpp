// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — compute (TRISC0/1/2).
//
// Block schedule per image (op_design.md "Block schedule"), all helpers:
//   pass 1  stage_chunk + colsum   copy chunk -> fp32 scratch; reduce<SUM,REDUCE_COL,Accumulate> -> cb_colsum_rows
//           group_partial_rows      matmul_block (1xK)@(KxNg) membership M     -> cb_partial_rows
//           combine round 0 (root)  the SAME matmul body: (1xGT)@(GTxNg) inv_rows @ gather -> cb_stats_bcast
//                                   (in0 row 0 = 1/n, so the product is the mean) -> writer mcast -> cb_group_mean
//   pass 2  expand_rows             the SAME body: (1xNg)@(Ngx1) stat @ M^T slice   -> cb_mean_rows
//           centered_sq_chunk       eltwise_chain Sub(bcast Row) -> Square     -> cb_fp32_scratch
//                                   (hw_mask programs — tile-rows partly outside their image, i.e. RM
//                                   shard heights that are stick counts or HW % 32 != 0 in any placement:
//                                   the SAME chain subtracts a FULL masked-mean tile expanded from the
//                                   writer's cb_masked_mean; head / body / tail segments each expand
//                                   their own mean operand)
//           colsum_accumulate_chunk reduce<...> with Accumulate (acc = cb_colsum_rows) -> cb_colsum_rows
//           group_partial_rows / combine round 1 (root, variance) -> mcast -> cb_group_var
//   two_pass (interleaved streaming without hw_mask): passes 1 and 2 above become ONE input
//           read — pass A per chunk: stage_chunk, [chunk 0: shift rows = inv32_row @ tile-row 0, the SAME
//           matmul body], colsum S, centered_sq against the shift, colsum U (behind S in the 2K cb_colsum_rows ring);
//           after round 0 two_pass_variance_rows turns (S, U, shift, expanded mean) into sum (x - mean)^2 per channel
//           and the group aggregation / combine round 1 run unchanged. Pass B is pass 3.
//   pass 3  finalize_rstd            eltwise_chain Copy -> +eps -> rsqrt on the Ng group tiles -> cb_group_rstd
//           expand rstd -> scale_rows (x gamma in place, FPU mul); expand mean -> shift
//           shift_full              eltwise_chain Mul, DestReuse Sub / Negative, unary_bcast<Row>
//           apply_chunk             eltwise_chain Mul(bcast Row) + DestReuse Add -> cb_output_tiles
//           [untilize_chunk]        untilize (ROW_MAJOR leg)
//
// Group aggregation is NEVER a tile reduce over lanes (groups straddle tile
// boundaries): the reduce side is `colsum_rows @ membership`, the apply side
// is the transposed membership matmul. Both mask sites read one CB.
//
// Placement: block-sharded inputs run the resident regime with the shard as the
// block. TILE shards: cb_input_tiles / cb_output_tiles are placed on the shard buffers, the
// reader pushes the Hs*K credits per image and every block offset is shard-absolute
// (row_off*K for image n's sub-block); the output pack lands at the same offsets — also when the
// output CB is a second index over the INPUT shard (in_place: each input tile is read for the
// last time in pass 3 before its slot is packed). RM shards: the reader stages the shard's sticks
// and this kernel tilizes image n's tile-rows once into the resident tiled CB (padded to Hs*K).
// A core whose shard has no sticks in image n skips the three passes but, as root, still runs the
// two combine rounds (kernels/groupnorm_sc_N_1_HW_C_geometry.hpp is the shared geometry).
//
// CB quantum contract (see the reader): streaming chunks are a nominal Q*K
// pages on cb_input_tiles / cb_output_tiles / cb_fp32_scratch; the tail chunk
// pads to Q*K with unread pages so linear block indexing never straddles the
// ring boundary. The resident block is padded to Hmax*K per image.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/kernels/groupnorm_sc_N_1_HW_C_geometry.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// Chain block size: min(K, DEST_AUTO_LIMIT) tiles per DEST window. The chain walks each K-tile row
// in blocks of b with a valid-remainder tail at the row end (chain.inl: the `wt_base` loop is INSIDE
// the `ht` loop), so a block never crosses a row, and every chunked CB (cb_fp32_scratch Q*K,
// cb_output_tiles D*Q*K) is a whole number of rows — no block ever straddles a ring boundary
// whatever b is.
constexpr uint32_t chain_block_size(uint32_t k, uint32_t lim) { return k < lim ? k : lim; }

// In1 base-offset functor for matmul_block: read the membership slice of
// channel tile T (tiles T*Ng .. T*Ng+Ng-1) from the fronted region.
struct In1Base {
    uint32_t base;
    ALWI uint32_t operator()(uint32_t /*block*/) const { return base; }
};

// noinline: 8 call sites; every inlined CB op carries the --dev watcher asserts (code size).
__attribute__((noinline, noclone)) void pad_push(uint32_t cb, uint32_t n) {
    if (n > 0) {
        cb_reserve_back(cb, n);
        cb_push_back(cb, n);
    }
}

__attribute__((noinline, noclone)) void pad_pop(uint32_t cb, uint32_t n) {
    if (n > 0) {
        cb_wait_front(cb, n);
        cb_pop_front(cb, n);
    }
}

// colsum_accumulate_chunk: per-channel column sums of a q x K chunk, accumulated
// across chunks. The running raw partial sums live in the OUTPUT CB itself
// (cb_rows doubles as the Accumulate accumulator: every reload pops a tile before
// the pack of that output re-fills the slot, so K pages suffice); the last chunk
// finalizes in place. One template instantiation for all chunks (code size).
template <uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_rows>
__attribute__((noinline, noclone)) void colsum_accumulate_chunk(uint32_t q, uint32_t K, uint32_t chunk, bool last) {
    ckl::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_COL,
        cb_in,
        cb_scaler,
        cb_rows,
        ckl::ReduceInputPolicy::BulkWaitBulkPop,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ckl::ReduceAlgorithm::AccumulateViaAdd>(
        ckl::ReduceInputBlockShape::of(q, K),
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::Accumulate(ckl::AccumulationConfig::with_cb(cb_rows), chunk, last));
}

// row_matmul_block: out[0][n] = sum_kk sum_j in0_kk[0][j] * in1_(kk,n)[j][n]  — (1 x K') @ (K' x N), in1 tile
// (kk, n) at in1_base + kk*N + n. The ONE matmul body of the op (runtime CB ids; noinline+noclone — the
// compute binary has to fit the kernel-config ring in the --dev (watcher) build, and separate templated /
// transposed instantiations are several KB each). Four uses:
//   * group aggregation: in0 = cb_colsum_rows (K' = K), in1 = cb_membership (M, K x Ng)      -> cb_partial_rows
//   * root combine:      in0 = cb_inv_rows (K' = GT, row 0 = 1/n), in1 = cb_gather (GT x Ng) -> cb_stats_bcast
//   * expansions (x K):  in0 = group stat (K' = Ng), in1 = cb_membership_t slice T (Ng x 1)  -> row tile T
// in0 is WaitAndRetainOnLastBlock (waited, never popped here — the caller pops, or retains the stat across
// the K expansion calls); in1 is NoWaitNoPop (the caller owns its lifecycle).
__attribute__((noinline, noclone)) void row_matmul_block(
    uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out, uint32_t Kp, uint32_t N, uint32_t in1_base) {
    CircularBuffer in0(cb_in0), in1(cb_in1), out(cb_out);
    ckl::matmul_block<
        false,
        false,
        ckl::LastBlockTarget::Out,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndRetainOnLastBlock,
        ckl::InputPolicy::NoWaitNoPop,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        ckl::NoPostKBlock,
        0,
        ckl::NoKBlockInnerDimFn,
        ckl::NoIn0Source,
        In1Base>(
        in0, in1, out, out, ckl::MatmulBlockShape::of(1, N, 1, 1, Kp, 1), {}, {}, 0, 0, {}, {}, {}, In1Base{in1_base});
}

// group_partial_rows_block: partial[0][g] = sum_T sum_c colsum_T[0][c] * M_T[c][g]; pops the K colsum tiles.
ALWI void group_partial_rows_block(uint32_t cb_colsum, uint32_t cb_memb, uint32_t cb_partial, uint32_t K, uint32_t Ng) {
    row_matmul_block(cb_colsum, cb_memb, cb_partial, K, Ng, 0);
    cb_pop_front(cb_colsum, K);
}

// expand_rows_block: rows_T[0][c] = sum_g stat[0][g] * MT_T[g][c], T = 0..K-1 (in1 = transposed membership).
// The group stat (in0) is retained; the caller pops it in release_block.
ALWI void expand_rows_block(uint32_t cb_stat, uint32_t cb_memb_t, uint32_t cb_rows, uint32_t K, uint32_t Ng) {
    for (uint32_t T = 0; T < K; ++T) {
        row_matmul_block(cb_stat, cb_memb_t, cb_rows, Ng, 1, T * Ng);
    }
}

// tilize_rows / untilize_rows: one shared body per ROW_MAJOR (un)tilize (three tilize
// sites; code size). The helpers are always-inline, so the wrapper's optimize attribute
// governs their K-trip loops: no full unrolling (the program must fit the kernel-config
// ring in the --dev build).
template <uint32_t K, uint32_t cb_sticks, uint32_t cb_tiles>
__attribute__((noinline, noclone, optimize("no-unroll-loops"))) void tilize_rows(uint32_t rows) {
    ckl::tilize<K, cb_sticks, cb_tiles>(rows);
}

template <uint32_t K, uint32_t cb_tiles, uint32_t cb_sticks>
__attribute__((noinline, noclone, optimize("no-unroll-loops"))) void untilize_rows(uint32_t rows) {
    ckl::untilize<K, cb_tiles, cb_sticks>(rows);
}

// stage_chunk (pass 1): copy a q x K input chunk into the fp32 scratch so pass 1
// and pass 2 share ONE accumulate-reduce instantiation (the input-CB reduce
// variants — WaitUpfrontNoPop for resident, chunked for streaming — would be a second
// body per TRISC; the program must fit the kernel-config ring). In the resident regime
// the caller waits per chunk (cumulative) instead of for the whole block, overlapping
// pass 1 with the reader's fill.
template <bool resident, uint32_t b, uint32_t cb_in, uint32_t cb_scratch>
ALWI void stage_chunk(uint32_t q, uint32_t K, uint32_t base) {
    const auto shape = ckl::IterationShape::grid(q, K).block_size(b);
    constexpr auto out = ckl::output(cb_scratch, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    if constexpr (resident) {
        constexpr auto x = ckl::input(
            cb_in,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Enabled,
            ckl::TileAddressing::Offset);
        ckl::eltwise_chain(shape, ckl::CopyTile<x>{base}, ckl::PackTile<out>{});
    } else {
        constexpr auto x = ckl::input(
            cb_in, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
        ckl::eltwise_chain(shape, ckl::CopyTile<x>{}, ckl::PackTile<out>{});
    }
}

// centered_sq_chunk: scratch = (x - mean)^2 over a q x K chunk.
//   default : mean operand = the row-0 mean tile of each column, broadcast down the rows.
//   hw_mask : (tile-rows only partly inside their image: RM shard heights that are stick counts,
//             or HW % 32 != 0 in any placement) the mean operand is a FULL tile per column —
//             mean_full[r][c] = mask[r] * mean_c, produced by the SAME expansion matmul from a
//             writer-masked group-mean tile (rows outside the image zeroed) — so the zero pad
//             sticks give (0 - 0)^2 = 0 and no mask multiply is needed. Exactly ONE chain per
//             binary either way (a separate masked chain or a fused mask-multiply element would
//             not fit the kernel-config ring in the --dev build). The x operand follows the
//             regime: resident blocks are addressed by offset (no wait / pop), streaming chunks
//             are waited / popped per block.
template <bool resident, bool hw_mask, uint32_t b, uint32_t cb_in, uint32_t cb_mean_rows, uint32_t cb_scratch>
ALWI void centered_sq_chunk(uint32_t q, uint32_t K, uint32_t base) {
    const auto shape = ckl::IterationShape::grid(q, K).block_size(b);
    constexpr auto out = ckl::output(cb_scratch, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto mean = ckl::input(
        cb_mean_rows,
        hw_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row);
    if constexpr (resident) {
        constexpr auto x = ckl::input(
            cb_in,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Enabled,
            ckl::TileAddressing::Offset);
        ckl::eltwise_chain(
            shape, ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, x, mean>{base}, ckl::Square<>{}, ckl::PackTile<out>{});
    } else {
        constexpr auto x = ckl::input(
            cb_in, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
        ckl::eltwise_chain(
            shape, ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, x, mean>{}, ckl::Square<>{}, ckl::PackTile<out>{});
    }
}

// pass2_segment: centered_sq_chunk + colsum_accumulate_chunk over `rows` tile-rows starting at
// `row_start`, in Q-row chunks. `chunk` is the running Accumulate
// index across segments, `total` the number of chunks in the whole pass (the last one finalizes).
// noinline+noclone: hw_mask programs call it from THREE sites (head / body / tail segments) and the
// always-inline chain must exist once per binary (code size: the kernel-config ring in the --dev
// build). The caller expands the segment's mean operand into
// cb_mean_rows before the call and pops it after.
template <
    bool resident,
    bool hw_mask,
    bool is_rm,
    uint32_t K,
    uint32_t Q,
    uint32_t b,
    uint32_t cb_in,
    uint32_t cb_sticks,
    uint32_t cb_mean_rows,
    uint32_t cb_scratch,
    uint32_t cb_scaler,
    uint32_t cb_rows>
__attribute__((noinline, noclone)) void pass2_segment(
    uint32_t row_start, uint32_t rows, uint32_t block_base, uint32_t& chunk, uint32_t total) {
    for (uint32_t r = row_start; r < row_start + rows; ++chunk) {
        const uint32_t q = (row_start + rows - r) < Q ? (row_start + rows - r) : Q;
        const uint32_t pad = (Q - q) * K;
        if constexpr (!resident && is_rm) {
            tilize_rows<K, cb_sticks, cb_in>(q);
            pad_push(cb_in, pad);
        }
        centered_sq_chunk<resident, hw_mask, b, cb_in, cb_mean_rows, cb_scratch>(q, K, block_base + r * K);
        pad_push(cb_scratch, pad);
        colsum_accumulate_chunk<cb_scratch, cb_scaler, cb_rows>(q, K, chunk, chunk + 1 == total);
        pad_pop(cb_scratch, pad);
        if constexpr (!resident) {
            pad_pop(cb_in, pad);
        }
        r += q;
    }
}

// root_combine_block: the root sums the gathered partial rows (x 1/n, the in0 row) with the SAME
// matmul body -> cb_stats_bcast (the writer multicasts it). One noinline body for both rounds.
__attribute__((noinline, noclone)) void root_combine_block(
    uint32_t cb_gather, uint32_t cb_inv_rows, uint32_t cb_stats_bcast, uint32_t GT, uint32_t Ng) {
    cb_wait_front(cb_gather, Ng * GT);
    group_partial_rows_block(cb_inv_rows, cb_gather, cb_stats_bcast, GT, Ng);  // pops the GT in0 tiles
    cb_pop_front(cb_gather, Ng * GT);
}

// apply_chunk: y = x * scale_row + shift_full over a q x K chunk; `fuse_silu` applies SiLU in DEST before the
// pack (SDXL: every resnet GroupNorm is followed by SiLU — fusing it saves one full pass over the activation).
template <
    bool resident,
    uint32_t b,
    uint32_t cb_in,
    uint32_t cb_scale_rows,
    uint32_t cb_shift_full,
    uint32_t cb_out,
    bool fuse_silu = false>
ALWI void apply_chunk(uint32_t q, uint32_t K, uint32_t base) {
    const auto shape = ckl::IterationShape::grid(q, K).block_size(b);
    constexpr auto scale_b = ckl::input(
        cb_scale_rows,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row);
    constexpr auto shift =
        ckl::input(cb_shift_full, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::InputTileMapping::Row);
    constexpr auto out = ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    if constexpr (resident) {
        constexpr auto x = ckl::input(
            cb_in,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Enabled,
            ckl::TileAddressing::Offset);
        if constexpr (fuse_silu) {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x, scale_b>{base},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, shift, ckl::DestReuseType::DEST_TO_SRCB>{},
                ckl::Silu<>{},
                ckl::PackTile<out>{});
        } else {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x, scale_b>{base},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, shift, ckl::DestReuseType::DEST_TO_SRCB>{},
                ckl::PackTile<out>{});
        }
    } else {
        constexpr auto x = ckl::input(
            cb_in, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
        if constexpr (fuse_silu) {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x, scale_b>{},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, shift, ckl::DestReuseType::DEST_TO_SRCB>{},
                ckl::Silu<>{},
                ckl::PackTile<out>{});
        } else {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x, scale_b>{},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, shift, ckl::DestReuseType::DEST_TO_SRCB>{},
                ckl::PackTile<out>{});
        }
    }
}

// two_pass_variance_rows: the per-channel centered second moment of this core's n sticks from
// the pass-A statistics, WITHOUT touching the input again. With S = sum x, U = sum (x - s)^2 (s = the shift the
// squares were taken against) and m = the group mean expanded to this channel:
//     sum_h (x - m)^2 = sum_h ((x - s) - (m - s))^2 = U + (m - s) (n (m + s) - 2 S) = U + 2 (s - m) R,
//     R = S - (n / 2) (m + s)
// Every operand is a row-0-valid fp32 tile: cb_colsum holds [S, U] (K tiles each, S in front), s sits in
// cb_shift, m in cb_mean_exp; v lands in cb_out_rows (K pages, empty before) — the in0 of the group
// aggregation matmul (runtime CB id). ONE chain over the K channel rows, once per image, two DEST slots:
//   D0 = m + s -> x(-n/2) -> +S = R          (FPU add, SFPU scalar, FPU DestReuse add)
//   D1 = s     -> m - D1  = m - s            (CopyTile into D1, DestReuse sub on D1)
//   D0 = D0 x D1 -> x(-2) -> +U = v          (binary SFPU multiply of the two slots, SFPU scalar, FPU add)
// Every input is held (caller-managed wait, no pop: S / U are read by index inside the 2K ring, U through
// TileAddressing::Offset base K) and popped after the chain; every DestReuse is DEST_TO_SRCB (CB (+,x) DEST —
// the scalars carry the signs). One chain instead of two and every operand past the first is fp32 without a
// reconfig emission (DataFormatReconfig::Disabled) — the compute program has to fit the kernel-config ring in
// the --dev build. Stability (the op's centered-variance contract): U is a sum of squares of shift-centered
// values and the correction a product of two O(sigma) differences of means, so no large term cancels
// against another; the only subtractions of nearly-equal quantities are the differences of means themselves,
// whose tf32 operand rounding (~2^-11 |mean|) is the same exposure the three-pass centered pass has. Never
// E[x^2] - mean^2.
template <uint32_t K, uint32_t cb_colsum, uint32_t cb_shift, uint32_t cb_mean_exp, uint32_t cb_out_rows>
__attribute__((noinline, noclone)) void two_pass_variance_rows(uint32_t neg_half_n_bits) {
    constexpr uint32_t NEG_TWO_F32_BITS = 0xC0000000u;
    constexpr auto none = ckl::WaitPolicy::None;
    constexpr auto nopop = ckl::PopPolicy::None;
    constexpr auto block = ckl::InputTileMapping::Block;
    constexpr auto no_reconfig = ckl::DataFormatReconfig::Disabled;
    constexpr auto m_in = ckl::input(cb_mean_exp, none, nopop, block);
    constexpr auto s_in_b = ckl::input(cb_shift, ckl::BroadcastDim::None, none, nopop, block);
    constexpr auto s_in = ckl::input(cb_shift, none, nopop, block, no_reconfig);
    constexpr auto m_in_nr = ckl::input(cb_mean_exp, none, nopop, block, no_reconfig);
    constexpr auto S_in = ckl::input(cb_colsum, none, nopop, block, no_reconfig);
    constexpr auto U_in = ckl::input(cb_colsum, none, nopop, block, no_reconfig, ckl::TileAddressing::Offset);
    cb_wait_front(cb_mean_exp, K);
    cb_wait_front(cb_shift, K);
    cb_wait_front(cb_colsum, 2 * K);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(K),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Add, m_in, s_in_b>{},
        ckl::MulUnary<>{neg_half_n_bits},
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, S_in, ckl::DestReuseType::DEST_TO_SRCB>{},
        ckl::CopyTile<s_in, ckl::Dst::D1>{},
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Sub, m_in_nr, ckl::DestReuseType::DEST_TO_SRCB, ckl::Dst::D1>{},
        ckl::MulBinary<>{},
        ckl::MulUnary<>{NEG_TWO_F32_BITS},
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, U_in, ckl::DestReuseType::DEST_TO_SRCB>{K},
        ckl::PackTile<ckl::output(cb_out_rows)>{});
    pad_pop(cb_mean_exp, K);
    pad_pop(cb_shift, K);
    pad_pop(cb_colsum, 2 * K);
}

}  // namespace

void kernel_main() {
    // ---- compile-time knobs / CB ids ------------------------------------
    constexpr bool is_rm = get_compile_time_arg_val(0) == 1;
    constexpr bool input_resident = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t K = get_compile_time_arg_val(2);  // block_c_tiles
    constexpr uint32_t Q = get_compile_time_arg_val(3);  // chunk_hw_tiles
    constexpr bool has_gamma = get_compile_time_arg_val(4) == 1;
    constexpr bool has_beta = get_compile_time_arg_val(5) == 1;
    constexpr bool sharded = get_compile_time_arg_val(6) == 1;
    // hw_mask (host-derived): some core's tile-row is only partly inside its image (RM shard heights
    // that are not multiples of 32, N > 1 shards straddling images, HW % 32 != 0 in any placement).
    // Then the ONE pass-2 chain subtracts full masked-mean tiles per segment (writer-fed
    // cb_masked_mean, expanded here) — never a second chain instantiation (the program must fit
    // the kernel-config ring in the --dev build).
    constexpr bool hw_mask = get_compile_time_arg_val(7) == 1;
    // rm_direct (host-derived): a ROW_MAJOR block shard consumed IN PLACE as the row-major
    // block of width lcm(shard_w, 32) = K*32 elements (m sticks per block row): the shard itself is the
    // tilize's input CB and the output shard the untilize's output CB (tile-sized pages), and block
    // lane j is channel c0 + j % c_period. No stick staging, no stick write-back.
    [[maybe_unused]] constexpr bool rm_direct = get_compile_time_arg_val(8) == 1;
    // two_pass (host-derived): interleaved streaming programs without a ragged tile-row take
    // BOTH statistics from one read of each chunk (pass A) and apply in pass B — see pass_a_chunk /
    // two_pass_variance_rows above. Resident, sharded and hw_mask programs keep the three-pass schedule.
    constexpr bool two_pass = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(11);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(12);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(13);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(14);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(15);
    constexpr uint32_t cb_colsum_rows = get_compile_time_arg_val(16);
    constexpr uint32_t cb_partial_rows = get_compile_time_arg_val(17);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(18);
    constexpr uint32_t cb_group_mean = get_compile_time_arg_val(19);
    constexpr uint32_t cb_group_var = get_compile_time_arg_val(20);
    constexpr uint32_t cb_mean_rows = get_compile_time_arg_val(21);
    constexpr uint32_t cb_scale_rows = get_compile_time_arg_val(22);
    constexpr uint32_t cb_shift_full = get_compile_time_arg_val(23);
    constexpr uint32_t cb_fp32_scratch = get_compile_time_arg_val(24);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(25);
    constexpr uint32_t cb_output_sticks = get_compile_time_arg_val(26);
    constexpr uint32_t cb_stats_bcast = get_compile_time_arg_val(27);
    constexpr uint32_t cb_inv_rows = get_compile_time_arg_val(28);
    constexpr uint32_t cb_membership_t = get_compile_time_arg_val(29);
    constexpr uint32_t cb_group_rstd = get_compile_time_arg_val(30);
    [[maybe_unused]] constexpr uint32_t cb_masked_mean = get_compile_time_arg_val(31);
    [[maybe_unused]] constexpr uint32_t cb_inv32_row = get_compile_time_arg_val(32);  // two_pass: shift matmul in0
    constexpr bool fuse_silu = get_compile_time_arg_val(33) == 1;  // SDXL: SiLU fused into the pass-3 apply chain

    // Sharded inputs are always resident (the shard is the block).
    static_assert(!sharded || input_resident, "block-sharded placement runs the resident regime");
    // Two-pass statistics stream chunks of whole tile-rows (n = 32 sticks per row, no masked segments).
    static_assert(
        !two_pass || (!input_resident && !hw_mask && !sharded), "two_pass is the streaming, tile-row-aligned schedule");
    // TILE shards: block offsets are shard-absolute (image n's sub-block starts at row_off*K).
    constexpr bool shard_offsets = sharded && !is_rm;

    // ---- runtime args: grid-wide constants are COMMON, per-core geometry per core ----
    const uint32_t N = get_common_arg_val<uint32_t>(0);
    const uint32_t Ng = get_common_arg_val<uint32_t>(1);
    const uint32_t GT = get_common_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_common_arg_val<uint32_t>(3);
    const uint32_t Hmax = get_common_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t HW = get_common_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t hw_stride = get_common_arg_val<uint32_t>(6);  // stick pitch between images
    const uint32_t is_active = get_arg_val<uint32_t>(0);
    const uint32_t H_core = get_arg_val<uint32_t>(1);
    const bool is_root = get_arg_val<uint32_t>(2) == 1;
    [[maybe_unused]] const uint32_t s0 = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t sticks_valid = get_arg_val<uint32_t>(4);
    const uint32_t row_hi = get_arg_val<uint32_t>(5);  // valid sticks of this block's last tile-row (interleaved)
    // two_pass: -n/2 as fp32 bits, n = this core's sticks per image = 32*H_core (host-derived).
    [[maybe_unused]] const uint32_t neg_half_n_bits = get_arg_val<uint32_t>(6);

    if (is_active == 0) {
        return;
    }

    // Boot once: Reverse is mandatory for matmul_block (in0 -> SrcB, in1 -> SrcA);
    // every helper keeps its default INPUT_AND_OUTPUT reconfig and re-arms its formats.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_colsum_rows, cb_membership, cb_partial_rows);

    // K = 1: the chain walks the chunk's Q x 1 tiles as one flat sequence and every tile maps to column 0,
    // so DEST blocks may span tile-rows — up to DEST_AUTO_LIMIT tiles per DEST window instead of 1.
    constexpr uint32_t b =
        K == 1 ? chain_block_size(Q, ckl::DEST_AUTO_LIMIT) : chain_block_size(K, ckl::DEST_AUTO_LIMIT);

    // Program-lifetime constants (single wait, never popped): the weight rows here;
    // the two membership matrices (matmul in1, NoWaitNoPop — their lifecycle is ours)
    // are waited at FIRST USE below so the writer's build overlaps the pass-1 fill
    // instead of stalling compute at the top.
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma_rows, K);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta_rows, K);
    }

    bool first_work = true;  // membership waits happen at the first image with work
    for (uint32_t image = 0; image < N; ++image) {
        // Which tile-rows of my block belong to this image (all of them when interleaved).
        const auto w = sharded ? groupnorm_geometry::image_work_sharded(image, s0, sticks_valid, HW, hw_stride)
                               : groupnorm_geometry::image_work_interleaved(H_core, row_hi);
        const uint32_t H_n = w.rows;
        const uint32_t num_chunks = (H_n + Q - 1) / Q;
        const uint32_t resident_pad = (Hmax - H_n) * K;
        const uint32_t block_base = shard_offsets ? w.row_off * K : 0;

        // ================= pass A (two_pass): S and U from ONE read of each chunk =================
        // pass_a_chunk: the chunk sits at the front of cb_input_tiles (nominal Q*K pages) and is read TWICE
        // from L1 through offset addressing (no pop until the end):
        //   stage_chunk           copy -> cb_fp32_scratch (the fp32 in1 of the shift matmul / reduce input)
        //   (S and U share cb_colsum_rows, 2K pages, [S, U] order — see below)
        //   [chunk 0] shift rows  the ONE matmul body, (1 x 1) @ (1 x K): inv32_row @ tile-row 0 -> cb_mean_rows
        //                         = the per-channel column mean of the first 32 sticks, the shift s
        //   colsum_accumulate     S += column sums of the chunk                       (cb_colsum_rows)
        //   centered_sq_chunk     (x - s)^2 -> cb_fp32_scratch (the pass-2 chain, s as its Row operand)
        //   colsum_accumulate     U += column sums of the centered squares            (cb_colsum_rows, behind S)
        if constexpr (two_pass) {
            if (w.active) {
                for (uint32_t c = 0; c < num_chunks; ++c) {
                    const uint32_t q = (H_n - c * Q) < Q ? (H_n - c * Q) : Q;
                    const uint32_t pad = (Q - q) * K;
                    const bool last = c + 1 == num_chunks;
                    if constexpr (is_rm) {
                        tilize_rows<K, cb_input_sticks, cb_input_tiles>(q);
                        pad_push(cb_input_tiles, pad);
                    }
                    cb_wait_front(cb_input_tiles, Q * K);
                    stage_chunk<true, b, cb_input_tiles, cb_fp32_scratch>(q, K, 0);
                    pad_push(cb_fp32_scratch, pad);
                    if (c == 0) {
                        cb_wait_front(cb_fp32_scratch, Q * K);  // the matmul's in1 is caller-managed (NoWaitNoPop)
                        row_matmul_block(cb_inv32_row, cb_fp32_scratch, cb_mean_rows, 1, K, 0);
                    }
                    // S and U alternate on the ONE accumulate-reduce instantiation and the ONE 2K-page ring
                    // (cb_colsum_rows): each call reloads and pops ITS K partials from the front (the oldest)
                    // and packs the K new ones at the back, so the ring always reads [S, U].
                    colsum_accumulate_chunk<cb_fp32_scratch, cb_scaler, cb_colsum_rows>(q, K, c, last);
                    pad_pop(cb_fp32_scratch, pad);
                    centered_sq_chunk<true, false, b, cb_input_tiles, cb_mean_rows, cb_fp32_scratch>(q, K, 0);
                    pad_push(cb_fp32_scratch, pad);
                    colsum_accumulate_chunk<cb_fp32_scratch, cb_scaler, cb_colsum_rows>(q, K, c, last);
                    pad_pop(cb_fp32_scratch, pad);
                    pad_pop(cb_input_tiles, Q * K);  // the chunk's nominal pages (wait already satisfied)
                }
                if (first_work) {
                    cb_wait_front(cb_membership, K * Ng);  // first use (writer-built, program lifetime)
                }
                // S -> partial group sums; S is RETAINED (WaitAndRetain) for the variance combine below.
                row_matmul_block(cb_colsum_rows, cb_membership, cb_partial_rows, K, Ng, 0);
            }
        } else if (w.active) {
            // ================= pass 1: column sums -> group mean =================
            if constexpr (input_resident && is_rm) {
                tilize_rows<K, cb_input_sticks, cb_input_tiles>(H_n);
                pad_push(cb_input_tiles, resident_pad);
            }
            if constexpr (shard_offsets) {
                cb_wait_front(cb_input_tiles, Hmax * K);  // the reader's per-image credits for the whole shard
            }
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t q = (H_n - c * Q) < Q ? (H_n - c * Q) : Q;
                const uint32_t pad = (Q - q) * K;
                if constexpr (input_resident && !shard_offsets) {
                    cb_wait_front(cb_input_tiles, (c * Q + q) * K);  // cumulative: overlap the fill
                } else if constexpr (is_rm) {
                    tilize_rows<K, cb_input_sticks, cb_input_tiles>(q);
                    pad_push(cb_input_tiles, pad);
                }
                stage_chunk<input_resident, b, cb_input_tiles, cb_fp32_scratch>(q, K, block_base + c * Q * K);
                pad_push(cb_fp32_scratch, pad);
                colsum_accumulate_chunk<cb_fp32_scratch, cb_scaler, cb_colsum_rows>(q, K, c, c + 1 == num_chunks);
                pad_pop(cb_fp32_scratch, pad);
                if constexpr (!input_resident) {
                    pad_pop(cb_input_tiles, pad);
                }
            }
            if (first_work) {
                cb_wait_front(cb_membership, K * Ng);  // first use (writer-built, program lifetime)
            }
            group_partial_rows_block(cb_colsum_rows, cb_membership, cb_partial_rows, K, Ng);
        }
        // combine round 0: the root sums the gathered rows (x 1/n) with the same matmul
        // body; the writer lands the broadcast mean in cb_group_mean on every core.
        if (is_root) {
            root_combine_block(cb_gather, cb_inv_rows, cb_stats_bcast, GT, Ng);
        }

        if constexpr (two_pass) {
            // ============ two_pass: variance rows from (S, U, s) and the landed group mean ============
            if (w.active) {
                if (first_work) {
                    cb_wait_front(cb_membership_t, K * Ng);  // first use (writer-built, program lifetime)
                    first_work = false;
                }
                // m = the group mean expanded to this core's channel rows (cb_scale_rows is free until pass 3).
                expand_rows_block(cb_group_mean, cb_membership_t, cb_scale_rows, K, Ng);
                two_pass_variance_rows<K, cb_colsum_rows, cb_mean_rows, cb_scale_rows, cb_shift_full>(neg_half_n_bits);
                group_partial_rows_block(cb_shift_full, cb_membership, cb_partial_rows, K, Ng);  // v rows -> partials
            }
        } else if (w.active) {
            // ================= pass 2: centered squares -> variance -> rstd =====
            if (first_work) {
                cb_wait_front(cb_membership_t, K * Ng);  // first use (writer-built, program lifetime)
                first_work = false;
            }
            // Segments: [head row] + Q-row body chunks + [tail row] (geometry.hpp pass2_segments —
            // the streaming TILE reader chunks pass 2 in the same order). Without hw_mask there is
            // one segment (the body). With hw_mask (a first /
            // last tile-row partly outside the image) the head / tail segments expand their mean
            // operand from the writer's MASKED group-mean tiles (rows outside the image zero) so
            // the same chain yields (x - 0)^2 = 0 on the zero pad sticks.
            const auto seg = groupnorm_geometry::pass2_segments(w, hw_mask);
            const bool head = seg.head;
            const bool tail = seg.tail;
            const uint32_t body_start = seg.body_start;
            const uint32_t body_end = seg.body_end;
            const uint32_t body_chunks = (body_end - body_start + Q - 1) / Q;
            const uint32_t total = (head ? 1 : 0) + (tail ? 1 : 0) + body_chunks;
            uint32_t chunk = 0;
            // One noinline body for every segment (the chain must exist once per binary).
            constexpr auto segment = pass2_segment<
                input_resident,
                hw_mask,
                is_rm,
                K,
                Q,
                b,
                cb_input_tiles,
                cb_input_sticks,
                cb_mean_rows,
                cb_fp32_scratch,
                cb_scaler,
                cb_colsum_rows>;
            // (pops go through the noinline pad_pop — the waits are already satisfied; inlined pops
            // with their --dev asserts would grow the code past the kernel-config ring.)
            if constexpr (hw_mask) {
                if (head) {
                    expand_rows_block(cb_masked_mean, cb_membership_t, cb_mean_rows, K, Ng);
                    segment(0, 1, block_base, chunk, total);
                    pad_pop(cb_mean_rows, K);
                    pad_pop(cb_masked_mean, Ng);
                }
                if (body_chunks > 0) {
                    expand_rows_block(cb_group_mean, cb_membership_t, cb_mean_rows, K, Ng);
                    segment(body_start, body_end - body_start, block_base, chunk, total);
                    pad_pop(cb_mean_rows, K);
                }
                if (tail) {
                    expand_rows_block(cb_masked_mean, cb_membership_t, cb_mean_rows, K, Ng);
                    segment(H_n - 1, 1, block_base, chunk, total);
                    pad_pop(cb_mean_rows, K);
                    pad_pop(cb_masked_mean, Ng);
                }
            } else {
                expand_rows_block(cb_group_mean, cb_membership_t, cb_mean_rows, K, Ng);
                segment(0, H_n, block_base, chunk, total);
                pad_pop(cb_mean_rows, K);  // chain B operand was Upfront / no-pop
            }
            group_partial_rows_block(cb_colsum_rows, cb_membership, cb_partial_rows, K, Ng);
        }
        // combine round 1: the root sums the centered-square rows (x 1/n) = variance;
        // the broadcast lands it in cb_group_var on every core (writer-produced).
        if (is_root) {
            root_combine_block(cb_gather, cb_inv_rows, cb_stats_bcast, GT, Ng);
        }

        if (!w.active) {
            // No work in this image, but the writer still landed (and pushed) both stats here —
            // its reserve is the flow control that keeps the next broadcast off our previous
            // image's stats until we are done with them. Drain them.
            pad_pop(cb_group_mean, Ng);
            pad_pop(cb_group_var, Ng);
            continue;
        }

        // ================= pass 3: apply =====================================
        // finalize_rstd_block: rstd = rsqrt(var + eps) on the Ng group tiles (SFPU cost
        // scales with Ng, not K).
        // cb_group_var is writer-produced, so the result goes to a compute-owned CB.
        {
            constexpr auto var_in = ckl::input(cb_group_var);  // PerTile wait/pop: compute is its sole consumer
            constexpr auto rstd_out = ckl::output(cb_group_rstd);
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(Ng),
                ckl::CopyTile<var_in>{},
                ckl::AddUnary<>{eps_bits},
                ckl::Rsqrt<>{},
                ckl::PackTile<rstd_out>{});
        }
        expand_rows_block(cb_group_rstd, cb_membership_t, cb_scale_rows, K, Ng);
        if constexpr (has_gamma) {
            // scale_rows_block: scale = rstd_row * gamma_row (in place, both row-0 valid; FPU)
            ckl::mul<
                ckl::input(cb_scale_rows),
                ckl::input(
                    cb_gamma_rows,
                    ckl::BroadcastDim::None,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block),
                ckl::output(cb_scale_rows)>(ckl::IterationShape::tiles(K));
        }
        expand_rows_block(cb_group_mean, cb_membership_t, cb_mean_rows, K, Ng);
        // shift_full_block: shift = beta - mean*scale (or -mean*scale), in place on cb_mean_rows ...
        {
            constexpr auto mean_in = ckl::input(cb_mean_rows);
            constexpr auto scale_in = ckl::input(
                cb_scale_rows,
                ckl::BroadcastDim::None,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block);
            constexpr auto shift_out = ckl::output(cb_mean_rows);
            if constexpr (has_beta) {
                constexpr auto beta_in =
                    ckl::input(cb_beta_rows, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(K),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, mean_in, scale_in>{},
                    ckl::DestReuseBinary<ckl::BinaryFpuOp::Sub, beta_in, ckl::DestReuseType::DEST_TO_SRCB>{},
                    ckl::PackTile<shift_out>{});
            } else {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(K),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, mean_in, scale_in>{},
                    ckl::Negative<>{},
                    ckl::PackTile<shift_out>{});
            }
        }
        // ... then replicate row 0 down all 32 rows -> full tiles for the DestReuse operand.
        ckl::unary_bcast<ckl::BroadcastDim::Row, ckl::input(cb_mean_rows), ckl::output(cb_shift_full)>(
            ckl::IterationShape::tiles(K));

        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t q = (H_n - c * Q) < Q ? (H_n - c * Q) : Q;
            const uint32_t pad = (Q - q) * K;
            if constexpr (!input_resident && is_rm) {
                tilize_rows<K, cb_input_sticks, cb_input_tiles>(q);
                pad_push(cb_input_tiles, pad);
            }
            apply_chunk<input_resident, b, cb_input_tiles, cb_scale_rows, cb_shift_full, cb_output_tiles, fuse_silu>(
                q, K, block_base + c * Q * K);
            if constexpr (is_rm) {
                untilize_rows<K, cb_output_tiles, cb_output_sticks>(q);
            } else if constexpr (!sharded) {
                pad_push(cb_output_tiles, pad);  // writer consumes nominal Q*K per chunk
            }
            // (TILE shard: the output CB is the output shard; the pack offsets are shard-absolute and
            // nobody consumes it, so no pad quantum.)
            if constexpr (!input_resident) {
                pad_pop(cb_input_tiles, pad);
            }
        }

        // ================= release_block =====================================
        if constexpr (input_resident) {
            pad_pop(cb_input_tiles, Hmax * K);  // whole padded block: pointer returns to base
        }
        // (pad_pop = wait + pop through ONE noinline body; every wait here is already satisfied — the
        // inlined pops each carried the --dev watcher asserts, and the program must fit the ring.)
        pad_pop(cb_group_mean, Ng);
        pad_pop(cb_group_rstd, Ng);  // cb_group_var was popped by the finalize chain
        pad_pop(cb_scale_rows, K);
        pad_pop(cb_shift_full, K);
    }
}
