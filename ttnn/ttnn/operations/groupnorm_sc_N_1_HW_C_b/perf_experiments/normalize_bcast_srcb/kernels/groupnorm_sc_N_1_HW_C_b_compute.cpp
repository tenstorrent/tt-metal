// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b compute.
//
// Per block (op_design.md "Block schedule"):
//   accumulate_moments  : [RM: tilize] -> per channel tile, elementwise fp32 DEST accumulators S += x - k and
//                         Q += x*(x - k) over every tile row of the window, one SFPU column collapse at the end
//                         (moments_rows, raw LLK) -> S1' = Sigma (x - k), T2 = Sigma x*(x - k) in the chan CBs
//   collapse            : matmul(chan row [1 x W], map [W x 1]) -> 2 local partial tiles (row 0 = group lanes)
//   combine (root)      : matmul(ones [1 x R], gathered [R x 2]) -> [S1', T2] replicated rows, finalized in the
//                         same DEST window (raw SFPU PostComputeFn, + the group shift k) -> [rstd, mean]
//   expand              : two matmuls against the reader's gamma map (+ beta map), raw LLK (expand_matmul):
//                         a = rstd x gmap, b' = [mean*rstd, e] x [gmap; bmap] = mean*rstd*gamma - beta, per
//                         channel (rows replicated); no affine: the group map, transposed
//   normalize           : per chunk y = (-1)*b' + x*a, two accumulating FPU multiplies into one DEST tile
//                         (raw compute API), resident chunks first; sharded RM packs untilized straight into the
//                         output shard (pack_untilize_dest)
// Shift k (Refinement 1): one sample per group, x[image row 0, first channel of the group], identical on every
// core of a unit, so the shifted sums combine with no correction. With mean' = S1'/n and T2 = Sigma x*(x - k),
//   var = T2/n - mean'*(mean' + k) = E[(x - k)^2] - mean'^2,
// whose cancellation scales with |k|*|mean - k| ~ |mean|*std instead of mean^2 (the one-pass E[x^2] - mean^2
// it replaces), and the 19-bit srcA/srcB truncation hits moments of that reduced size.
// Every helper keeps its default INPUT_AND_OUTPUT data-format reconfig; the one boot is the matmul
// contract compute_kernel_hw_startup<SrcOrder::Reverse>.

// Debug builds only (watcher / LLK asserts, i.e. --dev): the asserts inflate TRISC text, and GCC's IPA
// constant-propagation clones of the per-chunk wrappers then overflow the 70 688 B kernel-config ring on the RM
// root binary (measured up to 3.5 KB over). Production builds keep the clones: measured faster (sharded TILE
// vae_16384x512 70.0 us vs 78.0 us without them).
#if defined(WATCHER_ENABLED) || defined(WATCHER_NOINLINE) || defined(ENABLE_LLK_ASSERT)
#pragma GCC optimize("no-ipa-cp-clone")
#endif
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_output_rm = 1;  // sharded RM: the output shard (untilized rows)
// The input ring: one allocation, two FIFO views pushed in lockstep (alias group A). Pass 1 consumes every
// chunk through cb_input_tiles; cb_input_keep drops the streamed head chunks during pass 1 and holds the
// resident tail for pass 2, followed by the re-read head - exactly the pass-2 order.
constexpr uint32_t cb_input_tiles = 2;
constexpr uint32_t cb_input_keep = 3;
constexpr uint32_t cb_output_tiles = 5;
constexpr uint32_t cb_neg_one = 6;  // constant -1 (bf16, compute-generated once): pass 2's folded '- b''
constexpr uint32_t cb_sum_acc = 7;
constexpr uint32_t cb_sumsq_acc = 8;
constexpr uint32_t cb_chan_scale = 9;
constexpr uint32_t cb_chan_work = 10;
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;  // expand in1: gamma map (W tiles) + beta map (W, with beta)
constexpr uint32_t cb_local_partials = 15;
constexpr uint32_t cb_gathered_partials = 16;
constexpr uint32_t cb_ones_tiles = 17;
constexpr uint32_t cb_stats_out = 19;
constexpr uint32_t cb_group_stats = 20;
constexpr uint32_t cb_pilot_rows = 22;   // -k per channel (row 0), held for pass 1
constexpr uint32_t cb_group_shift = 24;  // root: k per group lane, rows replicated
// [mean*rstd, e] (Float32, K_B pages; e = 1.0 everywhere, column 0 is what counts), the b' expand matmul's in0. A view
// on the local partials' bytes (alias group D): the writer popped those after its gather write, before the stats
// multicast that the expand waits for could arrive, and the next collapse packs them only after this block's pass 2.
constexpr uint32_t cb_mr_e = 18;

constexpr uint32_t W = get_compile_time_arg_val(0);
constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);
constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
constexpr bool HAS_BETA = get_compile_time_arg_val(4) != 0;
constexpr uint32_t R = get_compile_time_arg_val(5);
// The input / output are L1 shards backing their CBs (TILE: the input ring and cb_output_tiles; RM: cb_input_rm
// and cb_output_rm). Sharded RM untilizes each normalized chunk into the output shard.
constexpr bool SHARDED = get_compile_time_arg_val(6) != 0;
constexpr bool UNTILIZE_OUT = SHARDED && IS_RM;
// One binary per role (the unit root combines and finalizes). Compile-time so the block loop is not unswitched on
// it (two copies of the whole per-block body would not fit the kernel-config ring under the watcher build).
constexpr bool IS_ROOT = get_compile_time_arg_val(7) != 0;
// Per-chunk device zones (perf_instrumentation.hpp): the host enables them only when every core's chunk count keeps
// the zone executions inside the per-RISC marker budget (a long DRAM block would silently exhaust it).
constexpr bool CHUNK_ZONES = get_compile_time_arg_val(8) != 0;
constexpr uint32_t T = CHUNK_HT * W;
// Tiles per DEST window of the pass-2 chain a*x - b': each element runs over the whole window, so its FPU init is paid
// once per window, not per tile. The chain clamps it to DEST.
constexpr uint32_t CHAIN_BLOCK = DEST_AUTO_LIMIT;

// b' lives in cb_chan_work (the expand's second matmul writes it there).
constexpr uint32_t cb_bprime = cb_chan_work;
// Expand matmul in1: the gamma map (+ beta map) from the reader, laid out [group lane x channel] (no transpose),
// else the plain group map, [channel x group lane] (unpacked transposed).
constexpr uint32_t cb_amap = HAS_GAMMA ? cb_gamma_rows : cb_group_map;
constexpr bool EXP_TRANSPOSE = !HAS_GAMMA;
constexpr uint32_t K_B = HAS_BETA ? 2 : 1;  // b' matmul inner tiles: [mean*rstd] (+ [e] against the beta map)
static_assert(
    !HAS_BETA || HAS_GAMMA, "the beta map rides in cb_gamma_rows behind the gamma map (beta-only is refused)");
// Expand matmul subblocking: every subblock is EXP_SBW_E = min(W, 4) tiles wide (one short init per product, a full
// fp32 half-sync DEST per subblock). W % 4 != 0: the last subblock is placed at column W - 4 (overlapping its
// predecessor) and packs only its trailing EXP_LAST tiles. So W = 5 runs 2 subblocks, not W / EXP_SBW = 5.
constexpr uint32_t EXP_SBW_E = W < 4 ? W : 4;
constexpr uint32_t EXP_NSB = (W + EXP_SBW_E - 1) / EXP_SBW_E;
constexpr uint32_t EXP_LAST = W - (EXP_NSB - 1) * EXP_SBW_E;  // tiles the last subblock publishes

// RM tilize is called from three sites (pass 1 / pre-tilize / pass 2) with a runtime extent; it is emitted
// once rather than inlined per site, so the five kernel binaries fit the kernel-config ring also under the
// watcher/assert (--dev) build. (The map matmuls stay inlined: measured, an out-of-line body grows the code.)
#define NOINLINE_BLOCK_OP __attribute__((noinline))

// Push (and for a CB-backed producer, reserve) the nominal-T pad of a ragged chunk.
ALWI void push_pad(uint32_t cb, uint32_t pad) {
    if (pad > 0) {
        cb_reserve_back(cb, pad);
        cb_push_back(cb, pad);
    }
}

// RM: tilize one chunk (rows_valid tile-rows of W tiles) from cb_input_rm into the input ring, then consume
// the rm pad pages and push the pad so every ring advances by the nominal T (both ring views in lockstep).
NOINLINE_BLOCK_OP void tilize_chunk(uint32_t rows_valid) {
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_tilize");
    DataflowBuffer rm_dfb(cb_input_rm);
    DataflowBuffer x_dfb(cb_input_tiles);
    cb_reserve_back(cb_input_keep, T);
    tilize<W, cb_input_rm, cb_input_tiles>(rm_dfb, x_dfb, rows_valid);
    const uint32_t pad = T - rows_valid * W;
    if (pad > 0) {
        cb_wait_front(cb_input_rm, pad);
        cb_pop_front(cb_input_rm, pad);
        push_pad(cb_input_tiles, pad);
    }
    cb_push_back(cb_input_keep, T);
}

// Advance one view of the input ring past a chunk it does not read.
ALWI void skip_chunk(uint32_t cb) {
    cb_wait_front(cb, T);
    cb_pop_front(cb, T);
}

// ---------------------------------------------------------------------------------------------------------------
// Pass-1 moments, "accumulate first, reduce once" (perf_experiments/pass1_moments_accumulate, candidate C2).
//
// RAW-LLK PATH (bypasses compute_kernel_lib::reduce<REDUCE_COL, Accumulate>, the eltwise_chain
// x*(x - k) -> Float32 intermediate CB and the dest-reuse helper; measured: unet_rm_ip_16384x960 167.8 -> 107.0 us,
// vae_tile_16384x512 54.6 -> 43.2 us, see changelog Perf 1 "Helper bypasses"). Mechanism: the FPU accumulates the two
// moments ELEMENTWISE in fp32 DEST over every tile row of the window; the 32 rows of each accumulator are collapsed
// once (SFPU column sum, fp32) at the end of the block:
//   S   += x + (-k)        ELWADD, acc_to_dest, srcB = the pilot row (-k) broadcast    (exact: bf16 operands)
//   D_i += x + (-k)        same MOP, into a zeroed DEST scratch tile -> d = x - k
//   Q   += x * d           ELWMUL (native DEST accumulate on WH/BH) at the user's HiFi3, srcA = x (unpacked),
//                          srcB = d moved DEST -> srcB (MOVD2B), exactly the baseline chain's x*(x - k) product
// so S ends at S1' = Sigma (x - k) and Q at T2 = Sigma x*(x - k), the moments the finalize already consumes.
// Why raw: the stock dest-reuse multiply (mul_reuse_dest_tiles) moves DEST[t] into srcB, ZEROACCs DEST[t] and
// writes the product back to the SAME tile t. Here the MOVD2B row offset reaches the scratch tile D_i while the
// MOP's destination counter walks the accumulator Q (no ZEROACC on Q: ELWMUL accumulates), and the ZEROACC clears
// D_i so the next tile's D_i += x + (-k) starts from 0. Same instructions, same count as the stock face loop.
// Keeping x*x out of the FPU matters: x*x at HiFi3 drops the srcA_lo x srcB_lo partial product (a ~1e-4 relative
// one-sided bias, which at |mean| >> std cancels catastrophically in T2 - mean'(mean' + k)); x*d with d = x - k
// a short-mantissa value on srcB is exact in that regime (candidate C1 measured that bias).
// No helper expresses "several elementwise FPU accumulators in one DEST window over a strided walk, then an SFPU
// column collapse": the reduce helper reloads/re-packs its accumulator per call, and the chain's DestAccumulation
// sums along a walk row, not down a column.
// DEST window (4 fp32 tiles, half-sync DEST_AUTO_LIMIT): one channel tile -> S (0), Q (1), D_0 (2), D_1 (3); rows
// are walked two at a time so each pair costs one ELWADD init and one multiply init.
// Across chunks (streamed DRAM / partially resident RM) the elementwise partials spill to the UnpackToDestFp32
// accumulators (CB 7 S, CB 8 Q; exact reload); the last chunk packs the collapsed row-0 moments straight into the
// FPU-readable chan CBs (9 / 10, the same bytes: in-place lockstep), which replaces the publish copy.
constexpr uint32_t S_DST = 0;
constexpr uint32_t Q_DST = 1;
constexpr uint32_t D0_DST = 2;
static_assert(DEST_AUTO_LIMIT >= 4, "C2 moments window needs 4 DEST tiles");

ALWI void add_bcast_rows_acc_init(uint32_t icb0, uint32_t icb1) {
    state_configure(icb0, icb1, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW, MathFidelity::LoFi>(
        icb0, icb1, 1 /*acc_to_dest*/)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

#ifdef TRISC_MATH
// Q += srcA * DEST[d]; DEST[d] := 0. Requires mul_reuse_dest_init<DEST_TO_SRCB> (HiFi3 face MOP + addr mods).
// Mirrors _llk_math_eltwise_binary_with_dest_reuse_<ELWMUL, NONE, ..., DEST_TO_SRCB> face by face.
template <uint32_t d>
inline void math_mul_acc_from_dest_tile() {
    static_assert(d > Q_DST && d < 4, "scratch tile must follow Q inside the 4-tile fp32 half");
    constexpr uint32_t off = (d - Q_DST) * 64;  // DEST rows per 32x32 tile slot (set_dst_write_addr stride)
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(Q_DST);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, off + 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, off + 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, off + 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, off + 12);
        // fp32 DEST: 4 tiles per bank, block index = tile * 4 + face (as the stock dest-reuse ZEROACC).
        TT_ZEROACC(p_zeroacc::CLR_16, 1 /*fp32*/, 0, ADDR_MOD_1, (d & 3) * 4 + f);
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}
#endif

template <uint32_t d>
ALWI void mul_x_by_dest_tile_acc(uint32_t x_cb, uint32_t x_idx) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(x_cb, x_idx)));
    MATH((math_mul_acc_from_dest_tile<d>()));
}

// rows tile rows of W tiles at the front of cb_input_tiles (row r, channel tile c at r*W + c).
NOINLINE_BLOCK_OP void moments_rows(uint32_t rows, bool first, bool last) {
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p1_moments");
    constexpr uint32_t x_cb = cb_input_tiles;
    pack_reconfig_data_format(cb_sum_acc);  // every target (7, 8, 9, 10) is Float32
    for (uint32_t c = 0; c < W; ++c) {
        tile_regs_acquire();
        if (!first) {
            cb_wait_front(cb_sum_acc, 1);
            cb_wait_front(cb_sumsq_acc, 1);
            reconfig_data_format_srca(cb_sum_acc);
            copy_init(cb_sum_acc);
            copy_tile(cb_sum_acc, 0, S_DST);
            copy_init(cb_sumsq_acc);
            copy_tile(cb_sumsq_acc, 0, Q_DST);
            cb_pop_front(cb_sum_acc, 1);
            cb_pop_front(cb_sumsq_acc, 1);
        }
        reconfig_data_format(x_cb, cb_pilot_rows);
        for (uint32_t r = 0; r < rows; r += 2) {
            const uint32_t t0 = r * W + c;
            const uint32_t t1 = t0 + W;
            const bool two = r + 1 < rows;
            add_bcast_rows_acc_init(x_cb, cb_pilot_rows);
            add_tiles_bcast<BroadcastType::ROW>(x_cb, cb_pilot_rows, t0, c, S_DST);
            add_tiles_bcast<BroadcastType::ROW>(x_cb, cb_pilot_rows, t0, c, D0_DST);
            if (two) {
                add_tiles_bcast<BroadcastType::ROW>(x_cb, cb_pilot_rows, t1, c, S_DST);
                add_tiles_bcast<BroadcastType::ROW>(x_cb, cb_pilot_rows, t1, c, D0_DST + 1);
            }
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(x_cb);
            mul_x_by_dest_tile_acc<D0_DST>(x_cb, t0);
            if (two) {
                mul_x_by_dest_tile_acc<D0_DST + 1>(x_cb, t1);
            }
        }
        if (last) {
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(S_DST);
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(Q_DST);
        }
        tile_regs_commit();
        const uint32_t s_cb = last ? cb_chan_scale : cb_sum_acc;
        const uint32_t q_cb = last ? cb_chan_work : cb_sumsq_acc;
        cb_reserve_back(s_cb, 1);
        cb_reserve_back(q_cb, 1);
        tile_regs_wait();
        pack_tile(S_DST, s_cb);
        pack_tile(Q_DST, q_cb);
        tile_regs_release();
        cb_push_back(s_cb, 1);
        cb_push_back(q_cb, 1);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// perf_experiments/normalize_bcast_srcb variants (selected by -D on every kernel; none = the op's n1u baseline).
//   GN_ABL_ROWB  : both multiplies take srcB as a ROW broadcast of one 16-datum row per face (row 0 of a / b' faces
//                  0 / 1) instead of a full 4 KB fp32 tile; srcA unchanged ((-1) tile, x).
//   GN_ABL_ZSUB  : DEST seeded with 0 - b' (ELWSUB ROW, srcA zeroed by the unpacker without an L1 read, srcB = one
//                  b' row per face), then DEST += x * a (ELWMUL ROW, HiFi3, srcB = one a row per face). The (-1)
//                  tile and its srcA unpack are gone. Inits per DEST window (stock MOPs).
//   GN_ABL_RAW   : with ZSUB, the seed is direct-issued (unpacker: UNPACR B row + UNPACR_NOP zero/SET_DVALID A per
//                  face; math: 2 ELWSUB + SETRWC per face), so the multiply's MOPs are programmed once per chunk.
//   GN_ABL_FULLB : srcB reads full 16-row faces (the stock ROW-broadcast B shape) instead of one row.
// Why it is exact: a and b' are rows-replicated (the expand matmul's output), so row 0 is the whole operand; the
// seed 0 - tf32(b') equals the baseline's HiFi3 (-1) * tf32(b') bit for bit (srcA = -1 has no low mantissa bits),
// and the x * a product and its accumulation into DEST are the same instructions in the same order.
#if defined(GN_ABL_ROWB) || defined(GN_ABL_ZSUB)
#define GN_P2_ROWB 1
#ifdef GN_ABL_FULLB
#define GN_ROWB_FR 16
#else
#define GN_ROWB_FR 1
#endif
#ifdef TRISC_UNPACK
// srcB shape of a / b': GN_ROWB_FR rows per face, 1x2 faces. Only the x-end (datums per UNPACR) changes; the L1 face
// stride is still the CB's full-tile tile-descriptor x_dim (256 datums), so UNPACR z+1 lands on face 1's row 0.
constexpr ckernel::TensorShape P2_ROWB_SHAPE = ckernel::make_tensor_shape(GN_ROWB_FR, 16, 1, 2);
#endif
ALWI void p2_mul_rowb_init(uint32_t a_cb, uint32_t b_cb) {
    state_configure(a_cb, b_cb, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MATH_FIDELITY>(a_cb, b_cb, 0)));
    UNPACK((_llk_unpack_AB_init_<BroadcastType::ROW>(
        ckernel::DEFAULT_TENSOR_SHAPE, P2_ROWB_SHAPE, ckernel::Transpose::None)));
}
ALWI void p2_mul_rowb(uint32_t a_cb, uint32_t b_cb, uint32_t ia, uint32_t ib, uint32_t dst) {
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(a_cb, b_cb, ia, ib)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(a_cb, b_cb, dst, true)));
}
#endif

#ifdef GN_ABL_ZSUB
#ifdef TRISC_UNPACK
// Per face: srcB = b' row (z += 1), srcA zeroed and handed to the FPU (no L1 read); B's z resets per face row.
static constexpr uint32_t P2_UNP_B_ROW =
    TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
static constexpr uint32_t P2_UNP_A_ZERO = TT_OP_UNPACR_NOP(
    SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
static constexpr uint32_t P2_UNP_B_Z0 = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001);
inline void p2_seed_unpack_mop() {
    ckernel_template tmp(2, 2, P2_UNP_B_ROW, P2_UNP_A_ZERO);
    tmp.set_end_op(P2_UNP_B_Z0);
    tmp.program();
}
// The seed's unpack without a MOP: _llk_unpack_A_<ROW, acc_to_dest> (B address only) with the MOP body inlined.
inline void p2_seed_unpack_raw(uint32_t b_cb, uint32_t ib) {
    const uint32_t id = get_operand_id(b_cb);
    const uint32_t address =
        get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * ib;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#ifdef GN_ABL_SEED2
    // Two source-bank handoffs per tile: (b' face-0 row, zero A) seeds DEST faces 0 and 2, (face-1 row, zero A)
    // seeds faces 1 and 3 (a row broadcast only reads the top faces' row 0).
    TTI_INSN(P2_UNP_B_ROW);
    TTI_INSN(P2_UNP_A_ZERO);
    TTI_INSN(P2_UNP_B_ROW);
    TTI_INSN(P2_UNP_A_ZERO);
    TTI_INSN(P2_UNP_B_Z0);
#else
#pragma GCC unroll 2
    for (uint32_t fr = 0; fr < 2; ++fr) {
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_ZERO);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_ZERO);
        TTI_INSN(P2_UNP_B_Z0);
    }
#endif
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif
#ifdef TRISC_MATH
// The seed's math without a MOP: per face 2 x ELWSUB (row broadcast, ADDR_MOD_0 = srcA +8 / srcB +0 / dest +8, as the
// ELWMUL ROW init left it) then flip both source banks, exactly the stock ELWSUB ROW MOP body.
inline void p2_seed_math_raw(uint32_t dst) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst);
#ifdef GN_ABL_SEED2
    // Bank pair 1 (b' face-0 row): DEST rows 0-15 (face 0), skip 16, rows 32-47 (face 2). srcA (zeroed) re-read
    // from row 0 for the second face.
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_A);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    // Flip both banks; DEST back to row 16. Bank pair 2 (b' face-1 row): rows 16-31 (face 1), rows 48-63 (face 3).
    TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_ABD);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_A);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_INCRWC(0, 8, 0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
#else
#pragma GCC unroll 4
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
    }
#endif
    math::clear_dst_reg_addr();
}
#endif
ALWI void p2_seed_init(uint32_t b_cb) {
    UNPACK((config_unpacker_x_end<p_setadc::UNP_B>(GN_ROWB_FR)));
    UNPACK((p2_seed_unpack_mop()));
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW, MATH_FIDELITY>(b_cb, b_cb, 0)));
}
// DEST[dst] = 0 - b'[ib] (row broadcast).
ALWI void p2_seed(uint32_t b_cb, uint32_t ib, uint32_t dst) {
#ifdef GN_ABL_RAW
    UNPACK((p2_seed_unpack_raw(b_cb, ib)));
    MATH((p2_seed_math_raw(dst)));
#else
    UNPACK((llk_unpack_A<BroadcastType::ROW, true>(b_cb, ib)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(b_cb, b_cb, dst, true)));
#endif
}
#endif

#if defined(GN_ABL_WSEED) || defined(GN_ABL_WMUL)
// GN_ABL_WSEED / GN_ABL_WMUL: one unpacker context (one base-address write, one semaphore round trip) per DEST window
// instead of per tile; each tile is addressed by the unpacker's ch0 Z counter (Z = 4 * tile + face, the L1 face
// stride is the tile descriptor's face size). b' / a are addressed from tile 0 of their CB (Z = 4 * w); x from the
// window's first tile.
#ifdef TRISC_UNPACK
static constexpr uint32_t P2_UNP_A_FACE =
    TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
inline uint32_t p2_cb_tile_addr(uint32_t cb, uint32_t tile) {
    const uint32_t id = get_operand_id(cb);
    return get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * tile;
}
inline void p2_seed_unpack_window(uint32_t cnt, uint32_t ww, uint32_t wrap_at, uint32_t reset_to) {
    const uint32_t address = p2_cb_tile_addr(cb_bprime, 0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t k = 0; k < cnt; ++k) {
        TT_SETADC(p_setadc::UNP_B, p_setadc::CH_0, p_setadc::SET_Z, ww << 2);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_ZERO);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_ZERO);
        ww = ww + 1 == wrap_at ? reset_to : ww + 1;
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
inline void p2_mul_unpack_window(
    uint32_t x_cb,
    uint32_t cnt,
    uint32_t xi0,
    uint32_t xrow_len,
    uint32_t xrow_skip,
    uint32_t ww,
    uint32_t wrap_at,
    uint32_t reset_to) {
    const uint32_t address_a = p2_cb_tile_addr(x_cb, xi0);
    const uint32_t address_b = p2_cb_tile_addr(cb_chan_scale, 0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    uint32_t rel = 0;
    uint32_t col = 0;
    for (uint32_t k = 0; k < cnt; ++k) {
        // The stock ROW MOP body: (B row, A face) x 2, B back to its face-0 row, (B row, A face) x 2.
        TT_SETADC(p_setadc::UNP_A, p_setadc::CH_0, p_setadc::SET_Z, rel << 2);
        TT_SETADC(p_setadc::UNP_B, p_setadc::CH_0, p_setadc::SET_Z, ww << 2);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_FACE);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_FACE);
        TT_SETADC(p_setadc::UNP_B, p_setadc::CH_0, p_setadc::SET_Z, ww << 2);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_FACE);
        TTI_INSN(P2_UNP_B_ROW);
        TTI_INSN(P2_UNP_A_FACE);
        ++rel;
        if (++col == xrow_len) {
            col = 0;
            rel += xrow_skip;
        }
        ww = ww + 1 == wrap_at ? reset_to : ww + 1;
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif
#ifdef TRISC_MATH
// Seed of DEST tiles 0..cnt-1 (two bank pairs per tile, as GN_ABL_SEED2).
inline void p2_seed_math_window(uint32_t cnt) {
    for (uint32_t k = 0; k < cnt; ++k) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(k);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_A);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_ABD);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_A);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_INCRWC(0, 8, 0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_ELWSUB(0, 0, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_ABD);
    }
    math::clear_dst_reg_addr();
}
inline void p2_mul_math_window(uint32_t x_cb, uint32_t cnt) {
    for (uint32_t k = 0; k < cnt; ++k) {
        llk_math_eltwise_binary<
            EltwiseBinaryType::ELWMUL,
            BroadcastType::ROW,
            DST_ACCUM_MODE,
            MATH_FIDELITY,
            EltwiseBinaryReuseDestType::NONE>(x_cb, cb_chan_scale, k, true);
    }
}
#endif
ALWI void p2_seed_window(uint32_t cnt, uint32_t w0, uint32_t wrap_at, uint32_t reset_to) {
    UNPACK((p2_seed_unpack_window(cnt, w0, wrap_at, reset_to)));
    MATH((p2_seed_math_window(cnt)));
}
ALWI void p2_mul_window(
    uint32_t x_cb,
    uint32_t cnt,
    uint32_t xi0,
    uint32_t xrow_len,
    uint32_t xrow_skip,
    uint32_t w0,
    uint32_t wrap_at,
    uint32_t reset_to) {
    UNPACK((p2_mul_unpack_window(x_cb, cnt, xi0, xrow_len, xrow_skip, w0, wrap_at, reset_to)));
    MATH((p2_mul_math_window(x_cb, cnt)));
}
#endif

// Per-chunk init and per-window seed / multiply of the pass-2 normalize, by variant (see the block above).
ALWI void p2_chunk_init(uint32_t x_cb) {
#if defined(GN_ABL_ZSUB)
    reconfig_data_format(x_cb, cb_bprime);  // srcA bf16 (x), srcB Float32 (b' / a)
#ifdef GN_ABL_RAW
    p2_mul_rowb_init(x_cb, cb_chan_scale);
#endif
#elif defined(GN_ABL_ROWB)
    reconfig_data_format(cb_neg_one, cb_bprime);
    p2_mul_rowb_init(cb_neg_one, cb_bprime);
#else
    reconfig_data_format(cb_neg_one, cb_bprime);
    mul_init(cb_neg_one, cb_bprime);
#endif
}
ALWI void p2_seed_window_init() {
#if defined(GN_ABL_ZSUB) && !defined(GN_ABL_RAW)
    p2_seed_init(cb_bprime);
#endif
}
ALWI void p2_seed_tile(uint32_t w, uint32_t dst) {
#if defined(GN_ABL_NOSEED)  // ablation only (wrong output): no seed
#elif defined(GN_ABL_ZSUB)
    p2_seed(cb_bprime, w, dst);
#elif defined(GN_ABL_ROWB)
    p2_mul_rowb(cb_neg_one, cb_bprime, 0, w, dst);
#else
    mul_tiles(cb_neg_one, cb_bprime, 0, w, dst);
#endif
}
ALWI void p2_mul_window_init(uint32_t x_cb) {
#if defined(GN_ABL_ZSUB) && !defined(GN_ABL_RAW)
    p2_mul_rowb_init(x_cb, cb_chan_scale);
#endif
}
ALWI void p2_mul_tile(uint32_t x_cb, uint32_t xi, uint32_t w, uint32_t dst) {
#if defined(GN_ABL_NOMUL)  // ablation only (wrong output): no multiply
#elif defined(GN_P2_ROWB)
    p2_mul_rowb(x_cb, cb_chan_scale, xi, w, dst);
#else
    mul_tiles(x_cb, cb_chan_scale, xi, w, dst);
#endif
}

// Pass-2 normalize of the chunk at the front of cb_input_keep: y = a*x - b' (a, b' row-mapped, held for the
// block), as two accumulating FPU multiplies into one DEST tile: DEST = (-1)*b' + x*a.
// perf_experiments/per_tile_pass_chains (N1b): raw compute API instead of eltwise_chain(BinaryFpu Mul +
// DestReuseBinary Sub); measured with N1u below: vae_tile_16384x512 54.6 -> 49.7 us, unet_rm_ip_16384x960
// 167.9 -> 153.9 us (changelog Perf 1, Helper bypasses). Mechanism: ELWMUL always accumulates into DEST, so the
// subtraction is folded into a second multiply by a constant -1 tile; both multiplies share one (srcA bf16, srcB
// Float32) format pair and one MOP, so there is no per-window re-init / reconfig and no DEST->src move (the dest-reuse
// MOVD2A stalls per face). Precision: b' and a sit on srcB (Float32 -> tf32, every bit used under HiFi3); -1 and x are
// bf16 on srcA (exact); the sum is formed in fp32 DEST (the dest-reuse path truncated a*x to tf32 on the DEST -> srcA
// move).
ALWI void normalize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    const uint32_t n = rows_valid * W;
    p2_chunk_init(x_cb);
    pack_reconfig_data_format(cb_output_tiles);
    uint32_t w = 0;
    for (uint32_t i = 0; i < n; i += CHAIN_BLOCK) {
        const uint32_t cnt = n - i < CHAIN_BLOCK ? n - i : CHAIN_BLOCK;
        cb_reserve_back(cb_output_tiles, cnt);
        tile_regs_acquire();
        uint32_t ww = w;
#ifdef GN_ABL_WSEED
        p2_seed_window(cnt, w, W, 0);
#else
        p2_seed_window_init();
        for (uint32_t j = 0; j < cnt; ++j) {
            p2_seed_tile(ww, j);
            ww = ww + 1 == W ? 0 : ww + 1;
        }
        ww = w;
#endif
#ifdef GN_ABL_WMUL
        p2_mul_window(x_cb, cnt, i, cnt, 0, w, W, 0);
        for (uint32_t j = 0; j < cnt; ++j) {
            ww = ww + 1 == W ? 0 : ww + 1;
        }
#else
        p2_mul_window_init(x_cb);
        for (uint32_t j = 0; j < cnt; ++j) {
            p2_mul_tile(x_cb, i + j, ww, j);
            ww = ww + 1 == W ? 0 : ww + 1;
        }
#endif
        w = ww;
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) {
            pack_tile(j, cb_output_tiles);
        }
        tile_regs_release();
        cb_push_back(cb_output_tiles, cnt);
    }
    push_pad(cb_output_tiles, T - n);
    cb_pop_front(x_cb, T);
}

// Sharded RM (UNTILIZE_OUT): the same folded normalize, packed straight to the row-major output shard with
// pack_untilize_dest (perf_experiments/per_tile_pass_chains N1u). Bypasses PackTile -> cb_output_tiles ->
// untilize<>() (unpack + datacopy + pack again per tile): the normalized DEST block is untilized by the packer.
// DEST block = UB_RT tile rows x UB_CT tile columns (UB_CT | W, UB_CT * UB_RT <= DEST window); the packer writes
// row r of the block at r * W pages and column block b at b * UB_CT tiles of the reserved tile rows.
constexpr uint32_t largest_divisor_le(uint32_t n, uint32_t m) {
    for (uint32_t b = m; b > 1; --b) {
        if (n % b == 0) {
            return b;
        }
    }
    return 1;
}
constexpr uint32_t UB_CT = largest_divisor_le(W, CHAIN_BLOCK);
constexpr uint32_t UB_RT = CHAIN_BLOCK / UB_CT;
constexpr uint32_t UB_NB = W / UB_CT;

ALWI void normalize_untilize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    p2_chunk_init(x_cb);
    pack_untilize_dest_init<UB_CT, W>(cb_output_rm);
    for (uint32_t r0 = 0; r0 < rows_valid; r0 += UB_RT) {
        const uint32_t rt = rows_valid - r0 < UB_RT ? rows_valid - r0 : UB_RT;
        cb_reserve_back(cb_output_rm, rt * W);
        for (uint32_t b = 0; b < UB_NB; ++b) {
            const uint32_t c0 = b * UB_CT;
            tile_regs_acquire();
#ifdef GN_ABL_WSEED
            p2_seed_window(rt * UB_CT, c0, c0 + UB_CT, c0);
#else
            p2_seed_window_init();
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < UB_CT; ++c) {
                    p2_seed_tile(c0 + c, r * UB_CT + c);
                }
            }
#endif
#ifdef GN_ABL_WMUL
            p2_mul_window(x_cb, rt * UB_CT, r0 * W + c0, UB_CT, W - UB_CT, c0, c0 + UB_CT, c0);
#else
            p2_mul_window_init(x_cb);
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < UB_CT; ++c) {
                    p2_mul_tile(x_cb, (r0 + r) * W + c0 + c, c0 + c, r * UB_CT + c);
                }
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<UB_CT, W>(cb_output_rm, rt, b);
            tile_regs_release();
        }
        cb_push_back(cb_output_rm, rt * W);
    }
    pack_untilize_uninit(cb_output_rm);
    cb_pop_front(x_cb, T);
}

// Collapse: [1 x W] per-channel row x the map [W x 1] (in1, resident, never popped) -> one tile.
ALWI void collapse_with_map(CircularBuffer& in0, CircularBuffer& map, CircularBuffer& out) {
    matmul_block<
        false,
        false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::NoWaitNoPop>(in0, map, out, out, MatmulBlockShape::of(1, 1, 1, 1, W, 1));
}

#ifdef TRISC_MATH
// One row-group-0 vector of the finalize (face F in {0, 1}, column parity P), then its results stored to every
// vector of the same parity in faces F and F + 2 (row-groups 0..3) of the two output tiles. DEST in: tile 0 = S1',
// tile 1 = T2 (the sender-sum matmul's rows are identical), tile 2 = k (the reader replicates its rows). DEST out:
// tile 0 = rstd, tile 1 = mean, every row. sfpi index = 8*face + 2*row_group + parity; a tile is 32 indices.
// Each step is the stock op's arithmetic in the stock order; the intermediates go through DEST like the stock
// chain (store, reload) so the compiler cannot contract a mul + add into one SFPMAD (a different rounding).
// The stock chain's ops round to bf16 when DEST is 16-bit; this path reproduces them for the op's fp32 DEST only.
static_assert(DST_ACCUM_MODE, "gn_finalize_sfpu matches the stock finalize chain for fp32_dest_acc_en=True only");
// Register budget: the exact sqrt body needs most LREGs, so only `var` is live into it - inv_n / eps are loaded
// (SFPLOADI) where used and mean is parked in tile 2's slot (k is dead by then) until rstd is stored.
template <int F, int P>
sfpi_inline void gn_finalize_vector(const uint32_t inv_n_bits, const uint32_t eps_bits) {
    constexpr int i = 8 * F + P;
    {
        const sfpi::vFloat inv_n = ckernel::sfpu::Converter::as_float(inv_n_bits);
        sfpi::dst_reg[i] = sfpi::vFloat(sfpi::dst_reg[i]) * inv_n;            // mean' = S1' * inv_n   (MulUnary)
        sfpi::dst_reg[32 + i] = sfpi::vFloat(sfpi::dst_reg[32 + i]) * inv_n;  // T2 * inv_n           (MulUnary)
    }
    {
        const sfpi::vFloat mp = sfpi::dst_reg[i];
        const sfpi::vFloat mean = sfpi::vFloat(sfpi::dst_reg[64 + i]) + mp;  // mean = k + mean'      (AddBinary)
        sfpi::dst_reg[64 + i] = mean;
        sfpi::dst_reg[i] = mp * mean;  // mean' * mean         (MulBinary)
    }
    sfpi::vFloat var = sfpi::vFloat(sfpi::dst_reg[32 + i]) - sfpi::vFloat(sfpi::dst_reg[i]);  //    (SubBinary)
    v_if(var < 0.0f) { var = 0.0f; }  //                      (Relu)
    v_endif;
    var = var + sfpi::vFloat(ckernel::sfpu::Converter::as_float(eps_bits));  // (AddUnary)
    sfpi::dst_reg[i] = var;
    const sfpi::vFloat rstd = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(sfpi::dst_reg[i]);  // (Rsqrt)
#pragma GCC unroll 4
    for (int rg = 0; rg < 4; ++rg) {
        sfpi::dst_reg[8 * F + 2 * rg + P] = rstd;
        sfpi::dst_reg[8 * (F + 2) + 2 * rg + P] = rstd;
    }
    const sfpi::vFloat mean = sfpi::dst_reg[64 + i];
#pragma GCC unroll 4
    for (int rg = 0; rg < 4; ++rg) {
        sfpi::dst_reg[32 + 8 * F + 2 * rg + P] = mean;
        sfpi::dst_reg[32 + 8 * (F + 2) + 2 * rg + P] = mean;
    }
}

// The 4 row-group-0 vectors (faces 0/1 x parities). A vector only writes its own parity in faces F and F + 2 (and
// its own tile-2 slot), and reads only its own row-group-0 slots, so no vector clobbers another's unread input.
sfpi_inline void gn_finalize_sfpu(const uint32_t inv_n_bits, const uint32_t eps_bits) {
    gn_finalize_vector<0, 0>(inv_n_bits, eps_bits);
    gn_finalize_vector<0, 1>(inv_n_bits, eps_bits);
    gn_finalize_vector<1, 0>(inv_n_bits, eps_bits);
    gn_finalize_vector<1, 1>(inv_n_bits, eps_bits);
}
#endif

// matmul_block PostComputeFn of the sender sum: DST 0/1 hold [S1', T2] (rows replicated); k joins as DST 2 (exact
// UnpackToDestFp32 copy), then the scoped finalize leaves [rstd, mean] in DST 0/1 for the helper's pack.
// RAW SFPU: bypasses the eltwise_chain finalize (CopyTile x3 + 8 full-tile SFPU ops + 2 PackTile, each SFPU op over
// all 32 vectors) - the chain has no vector scope and cannot run inside the matmul's DEST window. Measured (root
// finalize zone, perf_experiments/combine_critical_path): unet_rm_ip_1024x1280 whole op 27.1 -> 24.3 us with the
// publish hand-off; see changelog Perf 1 "Helper bypasses".
struct FinalizeStats {
    uint32_t inv_n_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t /*out_subblock_num_tiles*/) const {
        cb_wait_front(cb_group_shift, 1);
        reconfig_data_format_srca(cb_group_shift);
        copy_init(cb_group_shift);
        copy_tile(cb_group_shift, 0, 2);
        rsqrt_tile_init();  // SFPU config + the exact rsqrt constants (vConstIntPrgm0, vConstFloatPrgm1/2)
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            gn_finalize_sfpu, 0, ckernel::VectorMode::None, inv_n_bits, eps_bits)));
        cb_pop_front(cb_group_shift, 1);
    }
};

// RAW LLK (bypasses compute_kernel_lib::matmul_block for the two expand products only; measured, expand stage
// unet_rm_ip_1024x1280 2.76 -> 1.29 us, whole op 27.1 -> 25.4 us; changelog Perf 1, Helper bypasses). Mechanism: (1)
// the helper packs every subblock it computes, so a W = 5 / 7 / 10 / 15 row can only be tiled by the largest divisor of
// W that fits DEST (1 / 1 / 2 / 3 tiles): one DEST handshake + pack per 1..3 tiles. Here the last subblock overlaps its
// predecessor and packs only its new tiles (pack_block from a DEST offset), so every subblock is a full 4-tile
// DEST half (W = 5: 2 subblocks instead of 5) with a single short init per product. (2) Under fp32_dest_acc_en
// the helper re-issues pack_reconfig_data_format on every subblock; here once per product.
template <uint32_t K>
ALWI void expand_matmul(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    reconfig_data_format(in1_cb, in0_cb);  // srcA = in1 (the map), srcB = in0
    PACK((pack_reconfig_data_format(out_cb)));
    ckernel::matmul_block_init(in0_cb, in1_cb, EXP_TRANSPOSE, EXP_SBW_E, 1, K);
    cb_wait_front(in0_cb, K);
    for (uint32_t sb = 0; sb < EXP_NSB; ++sb) {
        const bool last = sb == EXP_NSB - 1;
        const uint32_t n0 = last ? W - EXP_SBW_E : sb * EXP_SBW_E;
        const uint32_t n_out = last ? EXP_LAST : EXP_SBW_E;
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            ckernel::matmul_block(in0_cb, in1_cb, k, k * W + n0, 0, EXP_TRANSPOSE, EXP_SBW_E, 1, K);
        }
        tile_regs_commit();
        cb_reserve_back(out_cb, n_out);
        tile_regs_wait();
        pack_block(EXP_SBW_E - n_out, out_cb, n_out);
        tile_regs_release();
        cb_push_back(out_cb, n_out);
    }
}

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(arg++);
    const uint32_t eps_bits = get_arg_val<uint32_t>(arg++);

    CircularBuffer chan_scale_cb(cb_chan_scale);
    CircularBuffer chan_work_cb(cb_chan_work);
    CircularBuffer map_cb(cb_group_map);
    CircularBuffer local_partials_cb(cb_local_partials);
    CircularBuffer ones_cb(cb_ones_tiles);
    CircularBuffer gathered_cb(cb_gathered_partials);
    CircularBuffer group_stats_cb(cb_group_stats);
    CircularBuffer stats_out_cb(cb_stats_out);

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_chan_scale, cb_group_map, cb_local_partials);

    // The constant -1 tile (bf16, srcA of pass 2's folded subtraction), resident for the whole kernel.
#ifndef GN_ABL_ZSUB
    {
        DataflowBuffer neg_one_dfb(cb_neg_one);
        eltwise_chain(
            IterationShape::one_tile(), FillScalar<Dst::D0>{-1.0f}, PackTile<output(cb_neg_one)>{neg_one_dfb});
    }
#endif

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    auto rows_valid_of = [&](uint32_t j) {
        const uint32_t rows_left = rows_core - j * CHUNK_HT;
        return rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ---------------- accumulate_moments (pass 1) ----------------
        {
            MaybeDeviceZoneScope("c_pass1");
            // The pilot (-k) is built by the reader after its pass-1 credit pushes (sharded) / chunk 0 (DRAM): RM
            // tilizes first, and the wait sits right before the first moments walk.
            auto wait_pilot = [] {
                MaybeDeviceZoneScope("c_wait_pilot");
                cb_wait_front(cb_pilot_rows, W);
            };
            if (SHARDED && K == num_chunks) {
                // The whole shard is (or becomes) resident: one window-major walk over all rows_core tile rows, so
                // the accumulators never leave DEST (no spill / reload). RM: tilize every chunk first (the ring
                // holds them contiguously: only the last chunk can be ragged, and its pad sits at its end).
                if constexpr (IS_RM) {
                    for (uint32_t j = 0; j < num_chunks; ++j) {
                        tilize_chunk(rows_valid_of(j));
                    }
                }
                wait_pilot();
                cb_wait_front(cb_input_tiles, num_chunks * T);
                moments_rows(rows_core, true, true);
                cb_pop_front(cb_input_tiles, num_chunks * T);
            } else {
                for (uint32_t j = 0; j < num_chunks; ++j) {
                    const uint32_t rows_valid = rows_valid_of(j);
                    if constexpr (IS_RM) {
                        tilize_chunk(rows_valid);
                    }
                    {
                        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p1_wait_x");
                        cb_wait_front(cb_input_tiles, T);
                    }
                    if (j == 0) {
                        wait_pilot();
                    }
                    moments_rows(rows_valid, j == 0, j + 1 == num_chunks);
                    cb_pop_front(cb_input_tiles, T);
                    if (j < first_resident) {
                        skip_chunk(cb_input_keep);  // streamed head chunk: not held for pass 2
                    }
                }
            }
            cb_pop_front(cb_pilot_rows, W);
        }
        if (blk == 0) {
            // The map is resident for the whole kernel (matmul in1, NoWaitNoPop). It is first read here, so the reader
            // builds it in pass 1's shadow.
            MaybeDeviceZoneScope("c_wait_map");
            cb_wait_front(cb_group_map, W);
        }
        {
            MaybeDeviceZoneScope("c_collapse");
            // ---------------- collapse_to_group_partials ----------------
            // [1 x W] per-channel row x [W x 1] map -> one tile; row 0 holds the G_core group lanes.
            collapse_with_map(chan_scale_cb, map_cb, local_partials_cb);
            collapse_with_map(chan_work_cb, map_cb, local_partials_cb);
        }

        // RM stall shadow: tilize the first pass-2 re-read chunk before blocking on the combine.
        bool pretilized = false;
        if constexpr (IS_RM) {
            if (first_resident > 0) {
                tilize_chunk(rows_valid_of(0));
                pretilized = true;
            }
        }

        // ---------------- combine_group_stats (root) ----------------
        if constexpr (IS_ROOT) {
            {
                // The gather round trip: every unit core's partial row landed (writer semaphore wait).
                MaybeDeviceZoneScope("c_wait_gather");
                cb_wait_front(cb_gathered_partials, 2 * R);
            }
            MaybeDeviceZoneScope("c_finalize");
            // Sum the cpi sender rows: ones [1 x R] x gathered [R x 2] -> [S1', T2] in DST 0/1 (every row = the
            // total), finalized in the same DEST window (FinalizeStats) and packed as [rstd, mean] (rows replicated).
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                FinalizeStats>(
                ones_cb,
                gathered_cb,
                stats_out_cb,
                stats_out_cb,
                MatmulBlockShape::of(1, 1, 1, 2, R, 1),
                FinalizeStats{inv_n_bits, eps_bits});
        }

        // ---------------- expand_stats_to_channels ----------------
        {
            // The combine round trip seen from this core: [rstd, mean] multicast back from the unit root.
            MaybeDeviceZoneScope("c_wait_stats");
            cb_wait_front(cb_group_stats, 2);
        }
        if constexpr (HAS_GAMMA) {
            // The reader builds the gamma / beta maps in the combine's stall shadow.
            MaybeDeviceZoneScope("c_wait_affine");
            cb_wait_front(cb_gamma_rows, K_B * W);
        }
        {
            MaybeDeviceZoneScope("c_expand");
            // Two matmuls emit a and b' directly (no W-tile chains):
            //   a  = rstd x amap              amap = gamma map (gamma[c] at (g(c), c)) / the plain map, transposed
            //   b' = [mr, e] x [amap; bmap]   mr = mean*rstd per group lane; bmap row 0 = -beta[c], e = 1
            // so b' = mean*rstd*gamma - beta (no beta: K_B = 1, b' = mean*rstd*gamma). Rows replicated, as the
            // stats rows are.
            DataflowBuffer group_stats_dfb(cb_group_stats);
            DataflowBuffer mr_e_dfb(cb_mr_e);
            // mr = rstd * mean (srcA rstd, srcB mean), [e = 1.0] -> cb_mr_e.
            cb_reserve_back(cb_mr_e, K_B);
            if constexpr (HAS_BETA) {
                eltwise_chain(
                    IterationShape::one_tile(),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_group_stats,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Scalar,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_group_stats,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Scalar,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{group_stats_dfb, group_stats_dfb, 0, 1},
                    FillScalar<Dst::D1>{1.0f},
                    PackTile<
                        output(
                            cb_mr_e,
                            ReservePolicy::None,
                            PushPolicy::None,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{mr_e_dfb, 0},
                    PackTile<
                        output(
                            cb_mr_e,
                            ReservePolicy::None,
                            PushPolicy::None,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D1>{mr_e_dfb, 1});
            } else {
                eltwise_chain(
                    IterationShape::one_tile(),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_group_stats,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Scalar,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_group_stats,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Scalar,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{group_stats_dfb, group_stats_dfb, 0, 1},
                    PackTile<
                        output(
                            cb_mr_e,
                            ReservePolicy::None,
                            PushPolicy::None,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{mr_e_dfb, 0});
            }
            cb_push_back(cb_mr_e, K_B);
            // a = rstd x amap^T -> cb_chan_scale, then drop rstd + mean.
            expand_matmul<1>(cb_group_stats, cb_amap, cb_chan_scale);
            cb_pop_front(cb_group_stats, 2);
            // b' = [mr, e] x [amap; bmap]^T -> cb_chan_work, then drop [mr, e].
            expand_matmul<K_B>(cb_mr_e, cb_amap, cb_chan_work);
            cb_pop_front(cb_mr_e, K_B);
            if constexpr (HAS_GAMMA) {
                cb_pop_front(cb_gamma_rows, K_B * W);
            }
        }

        // ---------------- normalize_block (pass 2): resident chunks first, then re-reads ----------------
        // cb_input_keep already fronts the resident tail; the re-read head lands behind it.
        // In place (sharded, the output CBs are the input shard's bytes) needs no extra ordering: TILE packs tile i
        // over the tile i it just unpacked (unpack -> math -> pack per tile); RM untilizes a chunk over the rows it
        // was tilized from, which happened in pass 1 (resident) or in the tilize_chunk just above (re-read).
        MaybeDeviceZoneScope("c_pass2");
        if constexpr (UNTILIZE_OUT) {
            // The output shard is a FIFO in row order: skip the head region (re-read last), write the resident
            // tail, then wrap to the head. The writer drains every credit.
            push_pad(cb_output_rm, first_resident * T);
        }
        for (uint32_t jj = 0; jj < num_chunks; ++jj) {
            const bool reread = jj >= K;
            const uint32_t j = reread ? jj - K : first_resident + jj;
            const uint32_t rows_valid = rows_valid_of(j);
            if constexpr (IS_RM) {
                if (reread && !(pretilized && j == 0)) {
                    tilize_chunk(rows_valid);
                }
            }
            if constexpr (UNTILIZE_OUT) {
                normalize_untilize_chunk(rows_valid);
            } else {
                normalize_chunk(rows_valid);
            }
            if (reread) {
                skip_chunk(cb_input_tiles);  // the pass-1 view also carried the re-read
            }
        }
        cb_pop_front(cb_chan_scale, W);
        cb_pop_front(cb_bprime, W);
    }
}
