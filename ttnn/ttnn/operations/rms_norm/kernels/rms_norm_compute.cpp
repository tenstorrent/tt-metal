// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC0/1/2). Block operations (op_design.md -> Block Operation Realization):
//   tilize_x_block      (RM)   compute_kernel_lib::tilize<C>        cb_x_rm -> cb_x
//   sum_squares_block          raw: x*x DEST-accumulated (FPU) per tile-row, DEST_AUTO_LIMIT tile-rows per acquire
//                              (masked W tail folded first) -> cb_sumsq_tiles
//   collapse_sumsq_block       reduce<SUM, ROW, ReduceTile> (FPU, 1.0 scaler cb_ones) -> cb_inv_rms (k==1, + finalize)
//                                                                                  -> cb_partial_sumsq (k>1)
//   combine_partials_block     symmetric (k <= 8 interleaved): reduce<SUM, ROW, AccumulateViaAdd, Skip>(rows x k)
//                              + finalize -> cb_inv_rms
//   collapse_rows_to_row0      owner topology: matmul ones-row x sumsq^T (unpacker transpose) -> row-form partial
//   combine_rows_block         owner topology: reduce_tile COL over ceil(k/32) gather tiles + transpose_dest + finalize
//                              -> cb_inv_owned
//   normalize_block            eltwise_chain BinaryFpu(x * inv_rms, bcast Col) [-> DestReuse * gamma bcast Row]
//                              -> cb_out
//   untilize_out_block  (RM)   compute_kernel_lib::untilize<C>      cb_out -> cb_out_rm
//
// Raw API / raw LLK (documented exceptions):
//  * MulGammaRowBcast — compute_kernel_lib::DestReuseBinary has no BroadcastDim (its
//    init/exec call mul_reuse_dest_init / mul_reuse_dest_tiles, which hard-code BroadcastType::NONE), but the LLK
//    supports a ROW-broadcast CB operand on the dest-reuse multiply. Consuming gamma row-broadcast is what lets the
//    writer skip the gamma expand (local NoC copy stages + bfp exponent re-pattern) and keep only faces 0/1 per gamma
//    page.
//  * finalize (FinalizeInvRms): column-0-scoped SFPU body (x * 1/W + eps -> rsqrt over the 8 even-parity vectors of
//    faces 0/2 only, one DEST store) run through the reduce() post_reduce_op hook. The stock SFPU entry points scope
//    only by face (VectorMode) / ITERATIONS, never by column parity, and binop_with_scalar + rsqrt_tile are three
//    full-tile passes (32 vectors each) — catalog sfpu_tile_scope c_skip. Capability gap, not ergonomics.
//  * pass 1 (sum_squares_block): eltwise_chain's DestAccumulation::PerRow cannot fold the masked W-tail tile into
//    the DEST accumulation (x*mask bcast-row, then DEST-reuse * x) and packs one tile-row per acquire. Capability gap
//    for the tail; the batching is a measured ~1 % extra.
//  * the within-tile row collapse uses the reduce() helper's ReduceTile (FPU) datapath, not AccumulateViaAdd's
//    copy_tile + sfpu_reduce: at one input tile per output the SFPU row reduce costs ~200 ns per tile-row (BH p150b)
//    against ~55 ns for unpack + GAPOOL (perf_experiments/fused_rowsum_finalize).
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
// Combine pipeline (sharded TILE owner combine, combine_pipeline): the resident shard is cut into sub-blocks of
// block_row_tiles tile-rows and run as  front(0); per t: front(t+1), combine(t), apply(t-1); apply(last)  so the
// exchange of t overlaps front(t+1) and the inv_rms return of t overlaps apply(t-1) + front(t+2) + combine(t+1).
// x (shard-backed, published whole by the reader) is popped only at kernel end; sub-block t is addressed at tile
// t * block_row_tiles * x_row_pitch (row_base). Row owners rotate per sub-block: row r of t is owned by
// (t * s + r) % k.
//
// Owner combine (k > K_SYM_MAX_*, WIDTH / BLOCK_SHARDED): tile-row r of a block is owned by member r % k; this
// member combines only its owned rows (k gathered partials each) into cb_inv_owned, which the writer returns
// to every member's cb_inv_rms. The gathered slots of the rows it does not own (uniform count per block keeps
// the peers' slot addressing in lockstep) are popped unread.
//
// Sharded TILE W tail (RMS_MASK_TAIL): cb_x is the caller's shard and is never written, so the tail tile's
// padding columns are masked instead of zero-filled: pass 1 seeds each tile-row's DEST tile with
// (x_tail * mask_row0) * x_tail before the full W tiles accumulate on top (cb_tail_mask, 0/1 row mask).

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
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#ifdef RMS_ROW_PARTIAL
#include "api/compute/reduce.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/matmul.h"
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace rms_norm_cb;

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(1);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(2);
constexpr uint32_t x_block_depth = get_compile_time_arg_val(3);
constexpr bool combine_lookahead = get_compile_time_arg_val(4) != 0;
constexpr bool combine_owner = get_compile_time_arg_val(5) != 0;
#ifndef RMS_ROW_PARTIAL
static_assert(!combine_owner, "owner topology: partials always travel in row form (RMS_ROW_PARTIAL)");
#endif
constexpr bool combine_pipeline = get_compile_time_arg_val(6) != 0;
#if defined(RMS_GAMMA_CHUNK) || defined(RMS_RM)
static_assert(!combine_pipeline, "combine pipeline: shard-backed TILE with resident gamma only");
#endif

// rsqrt(sum * (1/W) + eps) on column 0 of DST[dst] — the reduce() post_reduce_op. Column 0 lives only in the
// EVEN-parity SFPU vectors of faces 0 and 2: VectorMode::C visits faces 0/2, the body visits offsets 0,2,4,6
// (dst_reg += 2). mul + add stay in fp32 LREGs (one DEST store instead of three).
#ifdef TRISC_MATH
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat y =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] * inv_w + eps);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1,3,..,15 - never column 0)
    }
}
#endif

struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();  // sqrt-body constants
        MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, inv_w_bits, eps_bits)));
    }
};

// ---- sum_squares_block: per tile-row, sum_j x[r][j]^2 accumulated in DEST, one tile per tile-row ----
// Up to DEST_AUTO_LIMIT tile-rows per DEST acquire (DEST[i] = tile-row r0 + i; ELWMUL accumulates into DEST, as
// eltwise_chain's DestAccumulation::PerRow does). full_w: unmasked W tiles; with_tail: tile full_w is the masked
// W tail (sharded), folded FIRST: DEST[i] = x_t * mask (bcast row 0), DEST[i] = DEST[i] * x_t (DEST reuse -> srcA).
// row_base: tile offset of the block's first tile-row from the cb_x front (combine pipeline: x is popped only at
// kernel end, so the wait is cumulative from the front).
template <uint32_t cb_x_blk>
ALWI void sum_squares_block(uint32_t rows, uint32_t full_w, [[maybe_unused]] bool with_tail, uint32_t row_base = 0) {
    {
        MaybeDeviceZoneScope("cmp_wait_x");  // starved on the reader (interleaved) / tilize (RM)
        cb_wait_front(cb_x_blk, row_base + rows * x_row_pitch);
    }
    MaybeDeviceZoneScope("cmp_pass1");  // sum x^2 (DEST-accumulated), one pack per tile-row
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    reconfig_data_format(cb_x_blk, cb_x_blk);
    pack_reconfig_data_format(cb_sumsq_tiles);
#ifdef RMS_MASK_TAIL
    if (with_tail) {
        cb_wait_front(cb_tail_mask, 1);
    } else
#endif
    {
        mul_init(cb_x_blk, cb_x_blk, true);
    }
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_reserve_back(cb_sumsq_tiles, n);
        tile_regs_acquire();
#ifdef RMS_MASK_TAIL
        if (with_tail) {
            reconfig_data_format_srcb(cb_x_blk, cb_tail_mask);
            mul_bcast_rows_init(cb_x_blk, cb_tail_mask);
            for (uint32_t i = 0; i < n; ++i) {
                mul_tiles_bcast_rows(cb_x_blk, cb_tail_mask, row_base + (r0 + i) * x_row_pitch + full_w, 0, i);
            }
            reconfig_data_format_srcb(cb_tail_mask, cb_x_blk);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x_blk);
            for (uint32_t i = 0; i < n; ++i) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_x_blk, row_base + (r0 + i) * x_row_pitch + full_w, i);
            }
            mul_init(cb_x_blk, cb_x_blk, true);
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = row_base + (r0 + i) * x_row_pitch;
            for (uint32_t j = 0; j < full_w; ++j) {
                mul_tiles(cb_x_blk, cb_x_blk, base + j, base + j, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_sumsq_tiles);
        }
        tile_regs_release();
        cb_push_back(cb_sumsq_tiles, n);
    }
}

// ---- collapse_sumsq_block: within-tile row collapse on the FPU (column 0 = row sum) [+ finalize] ----
template <uint32_t out_cb, typename PostOp = NoOp>
ALWI void collapse_sumsq_block(uint32_t rows, PostOp post_op) {
    MaybeDeviceZoneScope("cmp_collapse");  // within-tile row collapse [+ finalize]
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq_tiles,
        cb_ones,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op);
}

#ifdef RMS_ROW_PARTIAL
// ---- Row partial (owner topology, Perf 2 row_partial_mcast): partials travel as ONE ROW ----
// Raw compute API (documented helper bypass): compute_kernel_lib::reduce has no row-form output — its REDUCE_ROW
// leaves the sums in column 0 and its post_reduce_op cannot transpose (transpose_dest reprograms the FPU MOP / address
// mods the next reduce_tile depends on) — and no "REDUCE_COL, then DEST transpose, then finalize" owner combine. The
// member collapse is a matmul with an unpacker transpose (no helper covers it). Capability gap, not ergonomics.
constexpr uint32_t row_gather = RMS_ROW_GATHER;  // gather tiles per owned row: ceil(k / 32), member m -> row m % 32

// collapse_rows_to_row0: the within-tile row collapse of the owner topology, emitted in ROW form. One FPU matmul per
// tile-row: in0 = cb_ones (row 0 = 1.0 x 32) -> srcB, in1 = the sumsq tile -> srcA, transposed by the unpacker
// (matmul_init transpose flag), so DST row 0 = ones x sumsq^T = the 32 row sums (face 0 row 0 + face 1 row 0: two
// contiguous 16-datum face rows the writer ships as is). No DEST transpose: BH p150b, measured against REDUCE_ROW +
// transpose_dest (exact, but ~140 ns per tile-row on the math thread): WIDTH (2048,256) k = 8 35.5 -> 30.0 us,
// (128,256) k = 8 5.38 -> 4.99 us, (128,4096) k = 128 8.0 -> 7.6 us. Precision vs torch equal or better on every
// measured cell incl. LoFi (perf_experiments/row_partial_mcast).
ALWI void collapse_rows_to_row0(uint32_t rows) {
    MaybeDeviceZoneScope("cmp_collapse");
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    cb_wait_front(cb_ones, 1);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_wait_front(cb_sumsq_tiles, n);
        cb_reserve_back(cb_partial_sumsq, n);
        reconfig_data_format(cb_sumsq_tiles, cb_ones);  // srcA = in1 (sumsq), srcB = in0 (ones)
        pack_reconfig_data_format(cb_partial_sumsq);
        matmul_init(cb_ones, cb_sumsq_tiles, 1 /*transpose in1*/);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_ones, cb_sumsq_tiles, 0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_partial_sumsq);
        }
        tile_regs_release();
        cb_push_back(cb_partial_sumsq, n);
        cb_pop_front(cb_sumsq_tiles, n);
    }
}

// combine_rows_block: per owned row, column-reduce its ceil(k/32) gather tiles (reduce_tile COL, 1.0 scaler: row 0 =
// sum over the members), transpose row 0 -> column 0 in DEST, finalize column 0 (rsqrt(sum/W + eps)) -> out_cb.
template <uint32_t out_cb>
ALWI void combine_rows_block(uint32_t owned, FinalizeInvRms finalize) {
    {
        MaybeDeviceZoneScope("cmp_wait_gathered");  // starved on the partial exchange (writer / peers)
        cb_wait_front(cb_gathered_sumsq, owned * row_gather);
    }
    MaybeDeviceZoneScope("cmp_combine");
    cb_wait_front(cb_ones, 1);
    for (uint32_t j = 0; j < owned; ++j) {
        cb_reserve_back(out_cb, 1);
        reconfig_data_format(cb_gathered_sumsq, cb_ones);  // REDUCE_COL: data -> srcA, scaler -> srcB
        pack_reconfig_data_format(out_cb);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered_sumsq, cb_ones, out_cb);
        tile_regs_acquire();
        for (uint32_t g = 0; g < row_gather; ++g) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered_sumsq, cb_ones, j * row_gather + g, 0, 0);
        }
        reduce_uninit();
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(0);
        finalize(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
    cb_pop_front(cb_gathered_sumsq, owned * row_gather);
}
#endif

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
// The unpacker reads full faces (x_end = 256 datums) from page base and base + face bytes (format Z stride), page j
// at rd_ptr + j * page_size: with the compact cb_gamma stride (192 B < 2 faces) rows 1..15 of srcB carry other
// pages' bytes, which SRCB_BCAST_ROW never reads (bench proof: NaN-filled unwritten bytes, output bit-identical).
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

// kStridedOut: pack window tile (r, c) at page base_col + r * x_row_pitch + c of a caller-reserved cb_out (the
// shard-backed output is row-major by tile-row). Needed when a multi-row block is normalized in several column
// windows (streamed gamma): appending per-window would land the tiles window-major — wrong for rows > 1.
// row_base: tile offset of the block's first tile-row in cb_x (combine pipeline; the output is never offset —
// cb_out is appended in sub-block order).
template <uint32_t cb_x_blk, bool kStridedOut = false>
ALWI void normalize_window(
    uint32_t rows, uint32_t base_col, uint32_t width, [[maybe_unused]] uint32_t gamma_col, uint32_t row_base = 0) {
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
    if constexpr (kStridedOut) {
        eltwise_chain(
            IterationShape::grid(rows, width),
            XTimesInvRms{StridedTileRange{base_col, x_row_pitch}},
            TimesGamma{gamma_col},
            PackTile<output(
                cb_out, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Strided)>{
                StridedTileRange{base_col, x_row_pitch}});
    } else {
        eltwise_chain(
            IterationShape::grid(rows, width),
            XTimesInvRms{StridedTileRange{row_base + base_col, x_row_pitch}},
            TimesGamma{gamma_col},
            PackTile<output(cb_out)>{});
    }
#else
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{row_base + base_col, x_row_pitch}},
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
ALWI void sum_and_collapse(
    uint32_t rows, uint32_t core_w_tiles, PostOp post_op, bool owns_tail, uint32_t row_base = 0) {
    sum_squares_block<cb_x_blk>(rows, owns_tail ? core_w_tiles - 1 : core_w_tiles, owns_tail, row_base);
    collapse_sumsq_block<out_cb>(rows, post_op);
}

// pass 1 + collapse of one block; k == 1 finalizes into cb_inv_rms, k > 1 leaves the raw partial.
template <uint32_t cb_x_blk>
ALWI void reduce_front(
    uint32_t rows, uint32_t core_w_tiles, FinalizeInvRms finalize, bool owns_tail, uint32_t row_base = 0) {
    if constexpr (w_split_cores == 1) {
        sum_and_collapse<cb_x_blk, cb_inv_rms>(rows, core_w_tiles, finalize, owns_tail, row_base);
    } else {
#ifdef RMS_ROW_PARTIAL
        sum_squares_block<cb_x_blk>(rows, owns_tail ? core_w_tiles - 1 : core_w_tiles, owns_tail, row_base);
        collapse_rows_to_row0(rows);
#else
        sum_and_collapse<cb_x_blk, cb_partial_sumsq>(rows, core_w_tiles, NoOp{}, owns_tail, row_base);
#endif
    }
}

// k > 1: combine the gathered partials of this block into inv_rms.
// ofs: owner rotation of the block (row r owned by member (r + ofs) % k); this member's first owned row is
// r0 = (member_idx - ofs) mod k. ofs == 0 is the unrotated assignment.
ALWI void combine_block(uint32_t rows, uint32_t member_idx, FinalizeInvRms finalize, uint32_t ofs = 0) {
    if constexpr (combine_owner) {
        const uint32_t r0 = (member_idx + w_split_cores - ofs) % w_split_cores;
        const uint32_t owned = r0 < rows ? (rows - r0 + w_split_cores - 1) / w_split_cores : 0;
#ifdef RMS_ROW_PARTIAL
        constexpr uint32_t per_row = row_gather;
#else
        constexpr uint32_t per_row = w_split_cores;
#endif
        const uint32_t slots = ((rows + w_split_cores - 1) / w_split_cores) * per_row;
        if (owned > 0) {
#ifdef RMS_ROW_PARTIAL
            combine_rows_block<cb_inv_owned>(owned, finalize);
#else
            combine_partials_block<cb_inv_owned>(owned, finalize);
#endif
        }
        const uint32_t unread = slots - owned * per_row;
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
// row_base / pop_x: combine pipeline — sub-block t sits at row_base in cb_x, which is popped only at kernel end.
template <uint32_t cb_x_blk>
ALWI void apply_back(
    uint32_t rows,
    uint32_t norm_w,
    uint32_t chunks_this,
    [[maybe_unused]] uint32_t gamma_tiles,
    uint32_t row_base = 0,
    bool pop_x = true) {
    {
        MaybeDeviceZoneScope("cmp_wait_inv");  // owner combine: inv_rms returned by the row owners (writer)
        cb_wait_front(cb_inv_rms, rows);
    }
#if defined(RMS_HAS_GAMMA) && !defined(RMS_GAMMA_CHUNK)
    {
        MaybeDeviceZoneScope("cmp_wait_gamma");  // resident gamma slice (writer load / multicast)
        cb_wait_front(cb_gamma, gamma_tiles + RMS_GAMMA_TAIL_PAGES);  // + tail pages (compact cb_gamma)
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
        cb_wait_front(cb_gamma, rm_chunk_w_tiles + RMS_GAMMA_TAIL_PAGES);
        normalize_window<cb_x_blk>(rows, c * rm_chunk_w_tiles, rm_chunk_w_tiles, 0);
        cb_pop_front(cb_gamma, rm_chunk_w_tiles + RMS_GAMMA_TAIL_PAGES);
#else
        normalize_window<cb_x_blk>(rows, c * rm_chunk_w_tiles, rm_chunk_w_tiles, c * rm_chunk_w_tiles);
#endif
        // untilize init/uninit per chunk: the normalize chain between chunks reprograms unpack/math/pack.
        untilize<rm_chunk_w_tiles, cb_out, cb_out_rm>(rows);
    }
#elif defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
    // Streamed gamma: one G-tile chunk per window, re-streamed by the writer every block (sharded TILE only:
    // cb_out is the row-major output shard). A multi-row block packs each window strided into the reserved block
    // (Perf 1 fix, found by gamma_stream_l1: appending put window-major tiles into the row-major shard, PCC 0.11 on
    // (1,1,8320,4608) HEIGHT fp32 gamma); a single tile-row's windows already land in order with the plain pack.
    constexpr uint32_t G = RMS_GAMMA_CHUNK;
    const bool strided_out = rows > 1;
    if (strided_out) {
        cb_reserve_back(cb_out, rows * x_row_pitch);
    }
    for (uint32_t c0 = 0; c0 < norm_w; c0 += G) {
        const uint32_t n = (norm_w - c0) < G ? (norm_w - c0) : G;
        cb_wait_front(cb_gamma, G + RMS_GAMMA_TAIL_PAGES);
        if (strided_out) {
            normalize_window<cb_x_blk, true>(rows, c0, n, 0);
        } else {
            normalize_window<cb_x_blk>(rows, c0, n, 0);
        }
        cb_pop_front(cb_gamma, G + RMS_GAMMA_TAIL_PAGES);
    }
    if (strided_out) {
        cb_push_back(cb_out, rows * x_row_pitch);
    }
#else
    normalize_window<cb_x_blk>(rows, 0, norm_w, 0, row_base);
#endif
    if (pop_x) {
        cb_pop_front(cb_x_blk, rows * x_row_pitch);
    }
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
    if constexpr (combine_pipeline && w_split_cores > 1) {
        // Combine pipeline (see the header): front(t+1), combine(t), apply(t-1). The writer matches it with a
        // two-round exchange window (rms_norm_writer.cpp).
        const uint32_t s_rows = block_row_tiles;
        auto base_of = [&](uint32_t t) { return t * s_rows * x_row_pitch; };
        auto front = [&](uint32_t t) { reduce_front<cb_x>(rows_of(t), core_w_tiles, finalize, owns_tail, base_of(t)); };
        auto apply = [&](uint32_t t) {
            apply_back<cb_x>(rows_of(t), norm_w, chunks_this, gamma_tiles, base_of(t), false);
        };
        front(0);
        for (uint32_t t = 0; t < num_blocks; ++t) {
            if (t + 1 < num_blocks) {
                front(t + 1);
            }
            combine_block(rows_of(t), member_idx, finalize, (t * s_rows) % w_split_cores);
            if (t >= 1) {
                apply(t - 1);
            }
        }
        apply(num_blocks - 1);
        cb_pop_front(cb_x, base_of(num_blocks - 1) + rows_of(num_blocks - 1) * x_row_pitch);
    } else if constexpr (w_split_cores > 1 && combine_lookahead) {
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
    cb_pop_front(cb_gamma, gamma_tiles + RMS_GAMMA_TAIL_PAGES);
#endif
#ifdef RMS_MASK_TAIL
    if (owns_tail) {
        cb_pop_front(cb_tail_mask, 1);  // pass 1 waits the mask per block, pops it once here
    }
#endif
}
