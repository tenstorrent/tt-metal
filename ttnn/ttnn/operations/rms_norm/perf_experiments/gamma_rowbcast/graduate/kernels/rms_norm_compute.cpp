// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC0/1/2). Block operations (op_design.md -> Block Operation Realization):
//   tilize_x_block      (RM)   compute_kernel_lib::tilize<C>        cb_x_rm -> cb_x
//   sum_squares_block          eltwise_chain BinaryFpu(x*x, PerRow DEST accumulation) -> cb_sumsq_tiles
//   collapse_sumsq_block       reduce<SUM, ROW, AccumulateViaAdd, Collapse>  -> cb_inv_rms (k==1, + finalize)
//                                                                           -> cb_partial_sumsq (k>1)
//   combine_partials_block     reduce<SUM, ROW, AccumulateViaAdd, Skip>(rows x k) + finalize -> cb_inv_rms
//   normalize_block            eltwise_chain BinaryFpu(x * inv_rms, bcast Col) [-> DestReuse * gamma bcast Row] ->
//   cb_out untilize_out_block  (RM)   compute_kernel_lib::untilize<C>      cb_out -> cb_out_rm
//
// Raw LLK (documented exception): MulGammaRowBcast — compute_kernel_lib::DestReuseBinary has no BroadcastDim (its
// init/exec call mul_reuse_dest_init / mul_reuse_dest_tiles, which hard-code BroadcastType::NONE), but the LLK supports
// a ROW-broadcast CB operand on the dest-reuse multiply. Consuming gamma row-broadcast is what lets the writer skip the
// gamma expand (local NoC copy stages + bfp exponent re-pattern) and keep only faces 0/1 per gamma page.
// Raw API (documented exception): the finalize post-op (mul 1/W, add eps, rsqrt) runs on DST inside the
// reduce() helper's post_reduce_op hook — reduce() exposes that hook as a uint32_t dst callable, not a chain,
// so an eltwise_chain Rsqrt would need the reduced tile packed to L1 and re-read (extra CB + pass).
//
// cb_x indexing: a tile-row occupies x_row_pitch pages in cb_x (uniform across the program; the last W member
// may own fewer tiles, its gap pages are never read), so cb_x is addressed Strided{base, x_row_pitch}.
// x_block_depth == 2 is realized as two one-block CBs (cb_x / cb_x_alt, block parity) rather than one ring of
// two blocks: indexed CB reads cannot wrap the ring, and the combine lookahead reads block b+1 while block b
// is still resident.
//
// Combine lookahead (k > 1, x_block_depth == 2): pass 1 + collapse of block b+1 run before combine(b), so they
// overlap the partial all-gather of block b (the only stage that waits on other Tensix cores).
//
// Owner combine (k > K_SYM_MAX_*, WIDTH / BLOCK_SHARDED): tile-row r of a block is owned by member r % k; this
// member combines only its owned rows (k gathered partials each) into cb_inv_owned, which the writer returns
// to every member's cb_inv_rms. The gathered slots of the rows it does not own (uniform count per block keeps
// the peers' slot addressing in lockstep) are popped unread.
//
// Sharded TILE W tail (RMS_MASK_TAIL): cb_x is the caller's shard and is never written, so the tail tile's
// padding columns are masked instead of zero-filled: pass 1 of the full W tiles collapses into the reduce
// Accumulate CB (raw sum), pass 1 of the tail tile is folded in with the 0/1 row mask (only_partial) and the
// finalize runs on that last chunk.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#ifdef RMS_RM
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif
#include "rms_norm_common.hpp"

using namespace compute_kernel_lib;
using namespace rms_norm_cb;

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(1);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(2);
constexpr uint32_t x_block_depth = get_compile_time_arg_val(3);
constexpr bool combine_lookahead = get_compile_time_arg_val(4) != 0;
constexpr bool combine_owner = get_compile_time_arg_val(5) != 0;

// rsqrt(sum * (1/W) + eps) on DST[dst] — the reduce() post_reduce_op.
struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
    }
};

// ---- sum_squares_block: per tile-row, sum_j x[r][j]^2 accumulated in DEST, one fp32 tile per tile-row ----
// base_col / width: the tile-column window of the block walked (sharded W tail: full tiles, then the tail).
template <uint32_t cb_x_blk>
ALWI void sum_squares_block(uint32_t rows, uint32_t base_col, uint32_t width) {
    {
        MaybeDeviceZoneScope("cmp_wait_x");  // starved on the reader (interleaved) / tilize (RM)
        cb_wait_front(cb_x_blk, rows * x_row_pitch);
    }
    MaybeDeviceZoneScope("cmp_pass1");  // sum x^2 (DEST-accumulated), one pack per tile-row
    constexpr auto x_in = input(
        cb_x_blk,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    using Square = BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D0, DestAccumulation::PerRow>;
    using PackRowSum = PackTile<output(
        cb_sumsq_tiles,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    eltwise_chain(
        IterationShape::grid(rows, width),
        Square{StridedTileRange{base_col, x_row_pitch}, StridedTileRange{base_col, x_row_pitch}},
        PackRowSum{});
}

// ---- collapse_sumsq_block: within-tile row collapse (column 0 = row sum) [+ finalize] ----
// scaler_cb: cb_x (never read: no partial tile, default reload — must only be distinct) or, for the masked
// W tail, cb_tail_mask (the 0/1 row mask folded into the last tile).
template <uint32_t out_cb, uint32_t scaler_cb = cb_x, typename AccumulateT = NoAccumulation, typename PostOp = NoOp>
ALWI void collapse_sumsq_block(
    uint32_t rows,
    PostOp post_op,
    AccumulateT acc = AccumulateT{},
    ReducePartialScaler partial = ReducePartialScaler::none()) {
    MaybeDeviceZoneScope("cmp_collapse");  // within-tile row collapse [+ finalize]
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq_tiles,
        scaler_cb,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(
        ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), acc, post_op, partial);
}

// ---- combine_partials_block: sum the k gathered column-0 partials per tile-row + finalize ----
template <uint32_t out_cb = cb_inv_rms>
ALWI void combine_partials_block(uint32_t rows, FinalizeInvRms finalize) {
    {
        MaybeDeviceZoneScope("cmp_wait_gathered");  // starved on the partial exchange (writer / peers)
        cb_wait_front(cb_gathered_sumsq, rows * w_split_cores);
    }
    MaybeDeviceZoneScope("cmp_combine");
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered_sumsq,
        cb_x,  // scaler slot: never read (see above)
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Skip>(
        ReduceInputBlockShape::of(rows, w_split_cores),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        finalize);
}

// ---- normalize_block: x * inv_rms (bcast Col) [* gamma (bcast Row) via DEST reuse] -> cb_out ----
// base_col: first tile column of the window (RM: chunk start); width: tiles walked per tile-row;
// gamma_col: cb_gamma page of the window's first column (resident: base_col; streamed chunk: 0).
#ifdef RMS_HAS_GAMMA
// MulGammaRowBcast: DEST[d] *= row-broadcast(cb tile) — a DestReuseBinary<Mul, In, DEST_TO_SRCA> whose CB operand
// (srcB) is ROW-broadcast: llk_unpack_A<ROW, acc_to_dest, DEST_TO_SRCA> unpacks faces 0,1,0,1 of the gamma page into
// srcB, llk_math_eltwise_binary<ELWMUL, ROW, ..., DEST_TO_SRCA> issues ELWMUL with SRCB_BCAST_ROW on each DEST face
// moved to srcA. Only row 0 of faces 0/1 of a gamma page is read. Same FPU/unpack work per tile as the NONE path.
// Inherits every chain trait (waits, srcB reconfig fold, tile addressing) from DestReuseBinary; init/exec replaced.
template <InputSpec In>
struct MulGammaRowBcast : DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA> {
    using Base = DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA>;
    static constexpr uint32_t GCb = In.cb_id;
    constexpr MulGammaRowBcast() noexcept = default;
    constexpr explicit MulGammaRowBcast(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        state_configure(GCb, __builtin_LINE());
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, GCb)));
        MATH((llk_math_eltwise_binary_init<
              EltwiseBinaryType::ELWMUL,
              BroadcastType::ROW,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, 0)));
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx =
            compute_kernel_lib::tile_base_value<Base::Addressing>(this->tile_base) +
            compute_kernel_lib::detail::input_idx<Base::Mapping, Base::Wait, Base::Pop, Base::Addressing>(
                i_flat, ht, wt, this->row_stride);
        const uint32_t dst = to_u32(Base::DstSlot) + slot_offset;
        UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, in_idx)));
        MATH((llk_math_eltwise_binary<
              EltwiseBinaryType::ELWMUL,
              BroadcastType::ROW,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, dst, true)));
    }
};
#endif

template <uint32_t cb_x_blk>
ALWI void normalize_window(uint32_t rows, uint32_t base_col, uint32_t width, [[maybe_unused]] uint32_t gamma_col) {
    constexpr auto x_in = input(
        cb_x_blk,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto inv_rms_in =
        input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_rms_in>;
#ifdef RMS_HAS_GAMMA
    constexpr auto gamma_in = input(
        cb_gamma,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Row,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
    using TimesGamma = MulGammaRowBcast<gamma_in>;
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{base_col, x_row_pitch}},
        TimesGamma{gamma_col},
        PackTile<output(cb_out)>{});
#else
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{base_col, x_row_pitch}},
        PackTile<output(cb_out)>{});
#endif
}

#ifdef RMS_RM
// One tilize call per tile-row: chunks_this blocks of width C (each waits C stick-piece pages of cb_x_rm and
// pushes C tiles into cb_x). The data-format reconfig runs only on the call that also inits: a reconfig issued
// after fast_tilize_init clobbers the fast-tilize unpack config (later tile-rows come out with rows 1..31 zero).
template <tilize_config::InitUninitMode mode>
ALWI void tilize_row(uint32_t chunks_this) {
    constexpr bool inits =
        mode == tilize_config::InitUninitMode::InitAndUninit || mode == tilize_config::InitUninitMode::InitOnly;
    tilize<
        rm_chunk_w_tiles,
        cb_x_rm,
        cb_x,
        mode,
        tilize_config::WaitMode::WaitBlock,
        inits ? tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
              : tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        tilize_config::Fp32Mode::Fast>(chunks_this);
}

// tilize_x_block: tilize init/uninit once per block (InitOnly / Neither / UninitOnly across tile-rows); each
// tile-row is padded to x_row_pitch pages (gap pages pushed unwritten — only the last W member has a gap).
ALWI void tilize_x_block(uint32_t rows, uint32_t chunks_this) {
    MaybeDeviceZoneScope("cmp_tilize");  // occupancy: includes the waits on the reader's stick pieces
    const uint32_t gap = x_row_pitch - chunks_this * rm_chunk_w_tiles;
    for (uint32_t r = 0; r < rows; ++r) {
        if (rows == 1) {
            tilize_row<tilize_config::InitUninitMode::InitAndUninit>(chunks_this);
        } else if (r == 0) {
            tilize_row<tilize_config::InitUninitMode::InitOnly>(chunks_this);
        } else if (r + 1 == rows) {
            tilize_row<tilize_config::InitUninitMode::UninitOnly>(chunks_this);
        } else {
            tilize_row<tilize_config::InitUninitMode::Neither>(chunks_this);
        }
        if (gap > 0) {
            cb_reserve_back(cb_x, gap);
            cb_push_back(cb_x, gap);
        }
    }
}
#endif

// pass 1 + collapse of one block into out_cb (+ post_op).
template <uint32_t cb_x_blk, uint32_t out_cb, typename PostOp>
ALWI void sum_and_collapse(uint32_t rows, uint32_t core_w_tiles, PostOp post_op, [[maybe_unused]] bool owns_tail) {
#ifdef RMS_MASK_TAIL
    if (owns_tail) {
        const uint32_t full = core_w_tiles - 1;  // W tiles before the tail tile
        if (full > 0) {
            sum_squares_block<cb_x_blk>(rows, 0, full);
            collapse_sumsq_block<cb_sumsq_acc, cb_tail_mask>(rows, NoOp{}, Accumulate::at(cb_sumsq_acc, 0));
            sum_squares_block<cb_x_blk>(rows, full, 1);
            collapse_sumsq_block<out_cb, cb_tail_mask>(
                rows, post_op, Accumulate::at_last(cb_sumsq_acc, 1), ReducePartialScaler::only_partial());
        } else {
            sum_squares_block<cb_x_blk>(rows, 0, 1);
            collapse_sumsq_block<out_cb, cb_tail_mask>(
                rows, post_op, NoAccumulation{}, ReducePartialScaler::only_partial());
        }
        return;
    }
#endif
    sum_squares_block<cb_x_blk>(rows, 0, core_w_tiles);
    collapse_sumsq_block<out_cb>(rows, post_op);
}

// pass 1 + collapse of one block; k == 1 finalizes into cb_inv_rms, k > 1 leaves the raw partial.
template <uint32_t cb_x_blk>
ALWI void reduce_front(uint32_t rows, uint32_t core_w_tiles, FinalizeInvRms finalize, bool owns_tail) {
    if constexpr (w_split_cores == 1) {
        sum_and_collapse<cb_x_blk, cb_inv_rms>(rows, core_w_tiles, finalize, owns_tail);
    } else {
        sum_and_collapse<cb_x_blk, cb_partial_sumsq>(rows, core_w_tiles, NoOp{}, owns_tail);
    }
}

// k > 1: combine the gathered partials of this block into inv_rms.
ALWI void combine_block(uint32_t rows, uint32_t member_idx, FinalizeInvRms finalize) {
    if constexpr (combine_owner) {
        const uint32_t owned = member_idx < rows ? (rows - member_idx + w_split_cores - 1) / w_split_cores : 0;
        const uint32_t slots = ((rows + w_split_cores - 1) / w_split_cores) * w_split_cores;
        if (owned > 0) {
            combine_partials_block<cb_inv_owned>(owned, finalize);
        }
        const uint32_t unread = slots - owned * w_split_cores;
        if (unread > 0) {
            cb_wait_front(cb_gathered_sumsq, unread);
            cb_pop_front(cb_gathered_sumsq, unread);
        }
    } else {
        combine_partials_block(rows, finalize);
    }
}

// norm_w: tiles normalized per tile-row (interleaved: core_w_tiles; sharded TILE: the whole shard row, so the
// shard-backed cb_out advances by x_row_pitch per tile-row — padding tiles of a ragged shard are don't-care).
// gamma_tiles: cb_gamma pages (waited here, not at kernel start: the writer loads gamma while pass 1 runs; the
// wait is a no-op after the first block — gamma is popped only at kernel exit).
template <uint32_t cb_x_blk>
ALWI void apply_back(uint32_t rows, uint32_t norm_w, uint32_t chunks_this, [[maybe_unused]] uint32_t gamma_tiles) {
    {
        MaybeDeviceZoneScope("cmp_wait_inv");  // owner combine: inv_rms returned by the row owners (writer)
        cb_wait_front(cb_inv_rms, rows);
    }
#if defined(RMS_HAS_GAMMA) && !defined(RMS_GAMMA_CHUNK)
    {
        MaybeDeviceZoneScope("cmp_wait_gamma");  // resident gamma slice (writer load / multicast)
        cb_wait_front(cb_gamma, gamma_tiles);
    }
#endif
    // occupancy: pass 2 (+ RM untilize) incl. back-pressure from the writer on cb_out / cb_out_rm and, when
    // gamma is streamed, the per-chunk gamma waits
    MaybeDeviceZoneScope("cmp_pass2");
#ifdef RMS_RM
    for (uint32_t c = 0; c < chunks_this; ++c) {
#if defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
        // Streamed gamma: one C-tile chunk per chunk window (the writer streams G == C tiles per chunk).
        static_assert(RMS_GAMMA_CHUNK == rm_chunk_w_tiles, "RM streamed gamma chunk must equal the RM chunk");
        cb_wait_front(cb_gamma, rm_chunk_w_tiles);
        normalize_window<cb_x_blk>(rows, c * rm_chunk_w_tiles, rm_chunk_w_tiles, 0);
        cb_pop_front(cb_gamma, rm_chunk_w_tiles);
#else
        normalize_window<cb_x_blk>(rows, c * rm_chunk_w_tiles, rm_chunk_w_tiles, c * rm_chunk_w_tiles);
#endif
        // untilize init/uninit per chunk: the normalize chain between chunks reprograms unpack/math/pack.
        untilize<rm_chunk_w_tiles, cb_out, cb_out_rm>(rows);
    }
#elif defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
    // Streamed gamma: one G-tile chunk per window, re-streamed by the writer every block.
    constexpr uint32_t G = RMS_GAMMA_CHUNK;
    for (uint32_t c0 = 0; c0 < norm_w; c0 += G) {
        cb_wait_front(cb_gamma, G);
        normalize_window<cb_x_blk>(rows, c0, (norm_w - c0) < G ? (norm_w - c0) : G, 0);
        cb_pop_front(cb_gamma, G);
    }
#else
    normalize_window<cb_x_blk>(rows, 0, norm_w, 0);
#endif
    cb_pop_front(cb_x_blk, rows * x_row_pitch);
    cb_pop_front(cb_inv_rms, rows);
}

void kernel_main() {
#ifdef RMS_ABLATE_EMPTY  // perf probes only: fixed-cost floor (see the reader)
    return;
#endif
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(4);
    const FinalizeInvRms finalize{get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6)};
    [[maybe_unused]] const uint32_t member_idx = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t norm_w = get_arg_val<uint32_t>(8);
    const bool owns_tail = get_arg_val<uint32_t>(9) != 0;

#ifdef RMS_RM
    compute_kernel_hw_startup(cb_x_rm, cb_x_rm, cb_x);
#else
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
#endif

#ifdef RMS_RM
    [[maybe_unused]] const uint32_t gamma_tiles = chunks_this * rm_chunk_w_tiles;
#else
    [[maybe_unused]] const uint32_t gamma_tiles = core_w_tiles;
#endif

    auto rows_of = [&](uint32_t blk) { return (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles; };
    auto use_alt = [&](uint32_t blk) { return x_block_depth == 2 && (blk & 1); };

#ifdef RMS_RM
    // RM: x_block_depth == 1 (compute is cb_x's producer): strictly sequential per block.
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = rows_of(blk);
        tilize_x_block(rows, chunks_this);
        reduce_front<cb_x>(rows, core_w_tiles, finalize, owns_tail);
        if constexpr (w_split_cores > 1) {
            combine_block(rows, member_idx, finalize);
        }
        apply_back<cb_x>(rows, core_w_tiles, chunks_this, gamma_tiles);
    }
#else
    if constexpr (w_split_cores > 1 && combine_lookahead) {
        reduce_front<cb_x>(rows_of(0), core_w_tiles, finalize, owns_tail);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t rows = rows_of(blk);
            if (blk + 1 < num_blocks) {
                if (use_alt(blk + 1)) {
                    reduce_front<cb_x_alt>(rows_of(blk + 1), core_w_tiles, finalize, owns_tail);
                } else {
                    reduce_front<cb_x>(rows_of(blk + 1), core_w_tiles, finalize, owns_tail);
                }
            }
            combine_partials_block(rows, finalize);
            if (use_alt(blk)) {
                apply_back<cb_x_alt>(rows, norm_w, chunks_this, gamma_tiles);
            } else {
                apply_back<cb_x>(rows, norm_w, chunks_this, gamma_tiles);
            }
        }
    } else {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t rows = rows_of(blk);
            if (use_alt(blk)) {
                reduce_front<cb_x_alt>(rows, core_w_tiles, finalize, owns_tail);
                if constexpr (w_split_cores > 1) {
                    combine_block(rows, member_idx, finalize);
                }
                apply_back<cb_x_alt>(rows, norm_w, chunks_this, gamma_tiles);
            } else {
                reduce_front<cb_x>(rows, core_w_tiles, finalize, owns_tail);
                if constexpr (w_split_cores > 1) {
                    combine_block(rows, member_idx, finalize);
                }
                apply_back<cb_x>(rows, norm_w, chunks_this, gamma_tiles);
            }
        }
    }
#endif

#if defined(RMS_HAS_GAMMA) && !defined(RMS_GAMMA_CHUNK)
    cb_pop_front(cb_gamma, gamma_tiles);
#endif
#ifdef RMS_MASK_TAIL
    if (owns_tail) {
        cb_pop_front(cb_tail_mask, 1);  // reduce() waits the mask but never pops it
    }
#endif
}
