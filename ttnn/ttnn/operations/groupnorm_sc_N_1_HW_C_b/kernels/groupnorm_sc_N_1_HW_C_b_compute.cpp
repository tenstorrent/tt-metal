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
//   combine (root)      : finalize_raw (raw LLK): -k = (-1 tile) x [k row; zeros] into DEST 2 before the gather
//                         wait, then (-1 tile) x gathered [R x 2] -> [-S1', -T2] in DEST 0/1, finalized in the same
//                         DEST window (raw SFPU, -inv_n) -> [rstd, mean*rstd] (rows replicated)
//   expand              : two matmuls against the reader's gamma map (+ beta map), raw LLK (expand_matmul), in0
//                         read from cb_group_stats = [rstd, mean*rstd, e] (e = 1.0, written once by the writer):
//                         a = rstd x gmap, b' = [mean*rstd, e] x [gmap; bmap] = mean*rstd*gamma - beta, per
//                         channel (rows replicated); no affine: the group map, transposed
//   normalize           : per chunk y = x*a - b': DEST seeded with 0 - b' (raw LLK, one unpacker context per DEST
//                         window), then DEST += x*a (ELWMUL, srcB a row broadcast); resident chunks first; sharded
//                         RM packs untilized straight into the output shard (pack_untilize_dest)
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
constexpr uint32_t cb_neg_one = 6;  // constant -1 (bf16, compute-generated once): the root finalize's in0 (-k, -S)
constexpr uint32_t cb_sum_acc = 7;
constexpr uint32_t cb_sumsq_acc = 8;
constexpr uint32_t cb_chan_scale = 9;
constexpr uint32_t cb_chan_work = 10;
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;  // expand in1: gamma map (W tiles) + beta map (W, with beta)
constexpr uint32_t cb_local_partials = 15;
constexpr uint32_t cb_gathered_partials = 16;
constexpr uint32_t cb_stats_out = 19;
constexpr uint32_t cb_group_stats = 20;
constexpr uint32_t cb_pilot_rows = 22;   // -k per channel (row 0), held for pass 1
constexpr uint32_t cb_group_shift = 24;  // root: k per group lane in row 0, rows 1..31 zero (finalize_raw)
// Sharded RM (Refinement 3, alias group F): partial / image-mixed tile rows staged k-filled by the reader (tilized
// in pass 1), and their normalized rows (untilized in pass 2, copied back to the output shard by the writer).
constexpr uint32_t cb_rm_stage_in = 25;
constexpr uint32_t cb_rm_stage_out = 26;
// DRAM interleaved output split across both DM RISCs: per pass-2 chunk the packer takes whichever output CB has a free
// chunk slot - the writer's cb_output_tiles first, else the reader's cb_output_b - and names the chunk's pass-2 index
// on that consumer's sequence CB (one 16 B uint32 page per chunk; OUT_SEQ_END closes the block). Each RISC writes its
// share on its own NoC, so a core whose writer NoC is congested hands the excess to the reader's NoC.
constexpr uint32_t cb_output_b = 27;
constexpr uint32_t cb_out_seq_a = 28;  // -> writer
constexpr uint32_t cb_out_seq_b = 29;  // -> reader
constexpr uint32_t OUT_SEQ_END = 0xFFFFFFFFu;

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
// Images per statistic unit (sharded: an image band of the grid; interleaved: 1). cb_group_stats holds
// [rstd, mean*rstd] per image, then e.
constexpr uint32_t NU = get_compile_time_arg_val(9);
constexpr uint32_t P = get_compile_time_arg_val(10);  // MAP_PERIOD: channels per slice period (sharded: shard width)
// Sharded RM with a padded last core column: lanes past the slice's valid channels are zeroed after the column
// collapse (the shard's padding bytes are not data).
constexpr bool COL_MASK = get_compile_time_arg_val(11) != 0;
constexpr bool HAS_STAGE = get_compile_time_arg_val(12) != 0;
// Sharded, fully resident: the core's shard is walked as op words (descriptor _rm_core_ops / TILE segments), one
// statistic set per image segment.
constexpr bool OPS = get_compile_time_arg_val(13) != 0;
constexpr uint32_t OP_ROWS = 0xFFFF;
constexpr uint32_t OP_STAGED = 1u << 16;
constexpr uint32_t OP_SEG_END = 1u << 17;
constexpr uint32_t OP_SKIP = 1u << 18;
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

// Sharded ops path: one piece (rows tile rows) -> the input ring, both views in lockstep; a staged piece (one tile
// row) comes from cb_rm_stage_in, a full one from the shard (cb_input_rm).
NOINLINE_BLOCK_OP void tilize_piece(uint32_t rows, [[maybe_unused]] bool staged) {
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_tilize");
    DataflowBuffer x_dfb(cb_input_tiles);
    cb_reserve_back(cb_input_keep, rows * W);
    if constexpr (HAS_STAGE) {
        if (staged) {
            DataflowBuffer st_dfb(cb_rm_stage_in);
            tilize<W, cb_rm_stage_in, cb_input_tiles>(st_dfb, x_dfb, rows);
            cb_push_back(cb_input_keep, rows * W);
            return;
        }
    }
    DataflowBuffer rm_dfb(cb_input_rm);
    tilize<W, cb_input_rm, cb_input_tiles>(rm_dfb, x_dfb, rows);
    cb_push_back(cb_input_keep, rows * W);
}

// Advance one view of the input ring past a chunk it does not read.
ALWI void skip_chunk(uint32_t cb) {
    cb_wait_front(cb, T);
    cb_pop_front(cb, T);
}

// ---------------------------------------------------------------------------------------------------------------
// Pass-1 moments, "accumulate first, reduce once", streamed one x tile per FPU program
// (perf_experiments/pass1_moments_accumulate C2, then perf_experiments/moments_fpu_stream).
//
// RAW-LLK PATH. It bypasses compute_kernel_lib::reduce<REDUCE_COL, Accumulate>, the eltwise_chain
// x*(x - k) -> Float32 intermediate CB, the dest-reuse helper (mul_reuse_dest_tiles), and the stock
// add_tiles_bcast<ROW> / mul_reuse_dest_init face MOPs that C2 still used.
// Measured, whole op (device ns), vs C2 on the same build:
//   unet_rm_ip_16384x960  68.9 -> 50.6 us
//   vae_tile_16384x512    33.1 -> 23.0 us
//   unet_rm_ip_16384x320  30.5 -> 24.3 us
//   unet_rm_ip_1024x1280  13.2 -> 11.7 us
// Output is bit-identical. See changelog "Helper bypasses".
//
// Mechanism. The FPU accumulates the two moments ELEMENTWISE in fp32 DEST over every tile row of the window. The
// 32 rows of each accumulator are collapsed once (SFPU column sum, fp32) at the end of the block:
//   D   = x + (-k)        ELWADD, srcB = the pilot row (-k) broadcast, into a DEST scratch tile (AddDst = 0)
//   S  += x + (-k)        same operands, AddDst = 1                                  (exact: bf16 operands)
//   Q  += x * d           ELWMUL (native DEST accumulate on WH/BH) at the user's fidelity. srcA = x, the same
//                         unpacked bank. srcB = d, moved DEST -> srcB (MOVD2B, fp32 -> bf16-style srcB), exactly
//                         the baseline chain's x*(x - k) product.
// S ends at S1' = Sigma (x - k) and Q at T2 = Sigma x*(x - k), the moments the finalize already consumes.
//
// One unpack per x tile feeds all three ops. The unpacker moves the WHOLE x tile into one srcA bank (faces 0..3 ->
// rows 0..63) and pilot faces 0 / 1 into one srcB bank (-k cols 0..15 in row 0, cols 16..31 in row 16). It uses one
// UNPACR each with the full-tile x_end, the matmul unpack's layout. The FPU then keeps both banks for one tile
// program:
//   8 x (ELWADD D, ELWADD S)  bcast row 0 for faces 0 / 2, row 16 for faces 1 / 3
//   STALLWAIT + 16 x MOVD2B   D -> srcB rows 0..63 (one drain per tile)
//   MOP: phases x 8 ELWMUL    fidelity loop OUTER over the 8 row blocks, so an accumulating ELWMUL never
//                             re-reads a DEST block written < 5 cycles earlier (Dst.md "Instruction scheduling");
//                             the last multiply flips both banks back to the unpacker.
// C2 paid three costs that this program removes:
//   - 3 x unpacks + 2 pilot unpacks + 1 dummy srcB publish per tile;
//   - a ZEROACC of the scratch;
//   - per-row-pair ELWADD / ELWMUL re-inits.
// It also ran 2-block face programs that stalled on every fidelity phase (pass 1 of vae_tile_16384x512:
// 17.9 -> 7.8 us). Same instructions, operands and per-element accumulation order as C2, so the output is
// bit-identical.
//
// Why x*x stays out of the FPU: x*x at HiFi3 drops the srcA_lo x srcB_lo partial product. That is a ~1e-4 relative,
// one-sided bias, which at |mean| >> std cancels catastrophically in T2 - mean'(mean' + k). x*d, with d = x - k a
// short-mantissa value on srcB, is exact in that regime (candidate C1 measured that bias).
//
// Why no helper or stock op expresses it: each stock binary op unpacks its own operands per call and owns its MOP,
// so no API keeps one unpacked srcA / srcB bank across ELWADD + MOVD2B + ELWMUL. The stock dest-reuse multiply
// ZEROACCs its source and writes back to the same tile. The reduce helper reloads / re-packs its accumulator per
// call, and the chain's DestAccumulation sums along a walk row, not down a column.
//
// DEST window (4 fp32 tiles, half-sync DEST_AUTO_LIMIT): one channel tile, with S at slot 0, Q at slot 1 and the
// scratch D at slot 2. Across chunks (streamed DRAM / partially resident RM) the elementwise partials spill to the
// UnpackToDestFp32 accumulators (CB 7 S, CB 8 Q; exact reload). The last chunk packs the collapsed row-0 moments
// straight into the FPU-readable chan CBs (9 / 10, the same bytes: in-place lockstep), which replaces the publish
// copy.
constexpr uint32_t S_DST = 0;
constexpr uint32_t Q_DST = 1;
static_assert(DEST_AUTO_LIMIT >= 3, "moments window needs 3 fp32 DEST tiles (S, Q, scratch D)");
// DEST rows relative to slot S_DST (fp32: 64 rows per tile slot), used as instruction DstRow immediates.
constexpr uint32_t MOM_S_ROW = S_DST * 64;
constexpr uint32_t MOM_Q_ROW = Q_DST * 64;
constexpr uint32_t MOM_D_ROW = 2 * 64;
// Address-mod slots 0..5 (the SFPU owns 6 / 7); every later stock init reprograms the slots it uses.
constexpr uint32_t AM_MOM_NONE = 0;     // no change (D adds, moves)
constexpr uint32_t AM_MOM_A8 = 1;       // srcA += 8
constexpr uint32_t AM_MOM_A8_B16 = 2;   // srcA += 8, srcB += 16 (end of faces 0 / 2: bcast row 0 -> 16)
constexpr uint32_t AM_MOM_A8_BCLR = 3;  // srcA += 8, srcB = 0  (end of faces 1 / 3: bcast row 16 -> 0)
constexpr uint32_t AM_MOM_MUL = 4;      // srcA, srcB, Dst += 8
constexpr uint32_t AM_MOM_MUL_PH = 5;   // srcA = srcB = Dst = 0, fidelity += 1

#ifdef TRISC_MATH
constexpr uint32_t MOM_PHASES =
    ckernel::math::is_high_fidelity(MATH_FIDELITY) ? ckernel::to_underlying(MATH_FIDELITY) : 1;

inline void moments_math_program() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(AM_MOM_NONE);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(AM_MOM_A8);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 16}, .dest = {.incr = 0}}.set(AM_MOM_A8_B16);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0}}.set(AM_MOM_A8_BCLR);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(AM_MOM_MUL);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = 0, .clr = 1},
        .fidelity = {.incr = 1}}
        .set(AM_MOM_MUL_PH);
    // D / S adds for the 8 row blocks: 16 entries, the FPU half of the replay buffer (the SFPU convention owns
    // 16..31).
    load_replay_buf(0, 16, [] {
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 0);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8, MOM_S_ROW + 0);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 8);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8_B16, MOM_S_ROW + 8);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 16);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8, MOM_S_ROW + 16);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 24);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8_BCLR, MOM_S_ROW + 24);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 32);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8, MOM_S_ROW + 32);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 40);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8_B16, MOM_S_ROW + 40);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 48);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8, MOM_S_ROW + 48);
        TTI_ELWADD(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_ROW, AM_MOM_NONE, MOM_D_ROW + 56);
        TTI_ELWADD(p_elwise::CLR_NONE, 1, p_elwise::SRCB_BCAST_ROW, AM_MOM_A8_BCLR, MOM_S_ROW + 56);
    });
    // Q += x * D: outer = fidelity phases, inner = the tile's 8 row blocks; the very last multiply flips both banks.
    ckernel_template tmp(
        MOM_PHASES, 8, TT_OP_ELWMUL(p_elwise::CLR_NONE, 0, p_elwise::SRCB_NO_BCAST, AM_MOM_MUL, MOM_Q_ROW));
    tmp.set_last_inner_loop_instr(
        TT_OP_ELWMUL(p_elwise::CLR_NONE, 0, p_elwise::SRCB_NO_BCAST, AM_MOM_MUL_PH, MOM_Q_ROW));
    tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_elwise::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, AM_MOM_MUL_PH, MOM_Q_ROW));
    tmp.program();
}

inline void moments_math_tile() {
    lltt::replay(0, 16);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB);  // srcA ended at row 64
    // D must be in DEST before the moves read it (the stock dest-reuse path stalls the same way; once per tile).
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);
    TTI_MOVD2B(0, 0, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 0);
    TTI_MOVD2B(0, 4, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 4);
    TTI_MOVD2B(0, 8, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 8);
    TTI_MOVD2B(0, 12, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 12);
    TTI_MOVD2B(0, 16, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 16);
    TTI_MOVD2B(0, 20, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 20);
    TTI_MOVD2B(0, 24, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 24);
    TTI_MOVD2B(0, 28, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 28);
    TTI_MOVD2B(0, 32, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 32);
    TTI_MOVD2B(0, 36, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 36);
    TTI_MOVD2B(0, 40, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 40);
    TTI_MOVD2B(0, 44, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 44);
    TTI_MOVD2B(0, 48, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 48);
    TTI_MOVD2B(0, 52, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 52);
    TTI_MOVD2B(0, 56, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 56);
    TTI_MOVD2B(0, 60, AM_MOM_NONE, p_movd2b::MOV_4_ROWS, MOM_D_ROW + 60);
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
#endif

#ifdef TRISC_UNPACK
// Per tile: one srcB UNPACR (pilot faces 0, 1 -> rows 0..31) and one srcA UNPACR (the whole x tile -> rows
// 0..63), each setting data-valid once. The tile descriptors stay as configured.
inline void moments_unpack_program() {
    TTI_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, 2 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    ckernel_template tmp(
        1,
        1,
        TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*SetDvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*SetDvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
#endif

// Unpacker + FPU state for the walk (once per DEST window): bf16 x on srcA, the pilot (bf16) on srcB.
ALWI void moments_init(uint32_t x_cb) {
    reconfig_data_format(x_cb, cb_pilot_rows);
    // Stock ELWADD-bcast init for its side state (CLR_DVALID enable, zero-flag default, RWC reset, ALU config); its
    // MOPs / x_end / addr mods are then superseded by the tile program.
    state_configure(x_cb, cb_pilot_rows, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW, MathFidelity::LoFi>(
        x_cb, cb_pilot_rows, 1 /*acc_to_dest*/)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(x_cb, cb_pilot_rows)));
    UNPACK((moments_unpack_program()));
    MATH((moments_math_program()));
    MATH((math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(S_DST)));
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F)));
}

#ifdef TRISC_MATH
// COL_MASK: after the column collapse, zero the S / Q lanes of channel tile c whose slice position is padding.
// Lane (face F, column parity PAR, vector lane j) is tile column F*16 + 2*(j & 7) + PAR; its slice position is
// (32c + column) mod P (sharded RM: k sticks of P channels per tile row), padding when >= vc. Only row-group 0 of
// faces 0 / 1 (rows 0..3: row 0 holds the sums) is touched, in both tiles (S at DEST 0, Q at DEST 1).
template <int F, int PAR>
sfpi_inline void col_mask_vector(const int base, const int vc) {
    const int b = base + F * 16 + PAR;
    sfpi::vInt pos = sfpi::vConstTileId & 14;
    pos = pos + sfpi::vInt(b);
    for (int lim = static_cast<int>(P); lim <= b + 14; lim += static_cast<int>(P)) {
        v_if(pos >= static_cast<int>(P)) { pos = pos - sfpi::vInt(static_cast<int>(P)); }
        v_endif;
    }
    v_if(pos >= sfpi::vInt(vc)) {
        sfpi::dst_reg[8 * F + PAR] = 0.0f;
        sfpi::dst_reg[32 + 8 * F + PAR] = 0.0f;
    }
    v_endif;
}
sfpi_inline void col_mask_sfpu(const int base, const int vc) {
    col_mask_vector<0, 0>(base, vc);
    col_mask_vector<0, 1>(base, vc);
    col_mask_vector<1, 0>(base, vc);
    col_mask_vector<1, 1>(base, vc);
}
#endif

// rows tile rows of W tiles from ring tile row row0 of cb_input_tiles (row r, channel tile c at (row0 + r)*W + c).
// Channel tiles c >= w_valid (a padded TILE slice) contribute nothing (their accumulators stay at DEST's zero).
NOINLINE_BLOCK_OP void moments_rows(
    uint32_t row0, uint32_t rows, bool first, bool last, uint32_t w_valid, [[maybe_unused]] uint32_t vc) {
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
        moments_init(x_cb);
        const uint32_t rows_c = c < w_valid ? rows : 0;
        for (uint32_t r = 0; r < rows_c; ++r) {
            UNPACK((llk_unpack_AB<BroadcastType::ROW>(x_cb, cb_pilot_rows, (row0 + r) * W + c, c)));
            MATH((moments_math_tile()));
        }
        MATH((math::clear_dst_reg_addr()));
        if (last) {
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(S_DST);
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(Q_DST);
            if constexpr (COL_MASK) {
                if (vc < P) {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(
                        col_mask_sfpu,
                        S_DST,
                        ckernel::VectorMode::None,
                        static_cast<int>((32 * c) % P),
                        static_cast<int>(vc))));
                }
            }
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

// Pass-2 normalize: DEST = (0 - b') + x * a per output tile, CHAIN_BLOCK tiles per DEST window.
// perf_experiments/normalize_bcast_srcb (Perf 2), measured against the (-1)*b' + x*a pair of mul_tiles it replaces
// (device ns, BH): vae_tile_16384x512 33136 -> 29505, unet_tile_4096x640_4x8 24623 -> 22404, unet_rm_ip_16384x960
// 68890 -> 66506, unet_tile_1024x1280 11597 -> 11068; bit-identical output on every case run.
// Helpers bypassed: mul_tiles / mul_init (compute API) for the seed, and llk_unpack_A / the binary MOP for its unpack.
// Mechanism:
//   - a and b' are rows-replicated (the expand matmul's output), so row 0 of each face is the whole operand: both
//     FPU ops take srcB as a ROW broadcast of one 16-datum row per face (x-end 16, tensor shape FR1 x NF1x2).
//   - The seed is ELWSUB(srcA = 0, srcB = b' row): the unpacker zeroes srcA without an L1 read (UNPACR_NOP
//     ZEROSRC + SET_DVALID), so the bf16 (-1) tile, its srcA unpack and two of the three HiFi3 phases are gone.
//     0 - tf32(b') == HiFi3 (-1) * tf32(b') bit for bit (srcA = -1 has no low mantissa bits).
//   - Measured (ablation): the seed's cost was the per-call unpacker context (base-address write, semaphore round
//     trip), not its data. One context covers the whole DEST window: b' tiles are addressed by the unpacker's ch0
//     Z counter (Z = 4 * w + face) from tile 0 of cb_bprime, and each tile takes 2 srcA/srcB bank pairs (the face-0
//     row seeds DEST faces 0 and 2, the face-1 row faces 1 and 3).
//   - The seed's math reuses the ELWMUL ROW init's ADDR_MOD_0 (srcA +8, srcB +0, dest +8), the stock ELWSUB ROW
//     addressing, so one init per chunk serves both ops; the x*a multiply is the stock per-tile call.
// History: Perf 1 (per_tile_pass_chains N1b / N1u) replaced eltwise_chain(BinaryFpu Mul + DestReuseBinary Sub)
// with two accumulating mul_tiles, (-1)*b' + x*a (no per-window re-init, no DEST->srcA move; changelog Perf 1).
// Precision: b' and a sit on srcB (Float32 -> tf32, every bit used under HiFi3), x is bf16 on srcA (exact); the sum
// is formed in fp32 DEST.
#ifdef TRISC_UNPACK
// srcB shape of a / b': one row per face, 1x2 faces. Only the x-end (datums per UNPACR) changes; the L1 face stride
// stays the CB's full-tile face size, so UNPACR z+1 lands on face 1's row 0.
constexpr ckernel::TensorShape P2_ROWB_SHAPE = ckernel::make_tensor_shape(1, 16, 1, 2);
static constexpr uint32_t P2_UNP_B_ROW =
    TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
static constexpr uint32_t P2_UNP_A_ZERO = TT_OP_UNPACR_NOP(
    SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
// Seed unpack of DEST tiles 0..cnt-1 from b' tiles ww, ww+1, ... (wrapping from wrap_at back to reset_to), in one
// unpacker context: _llk_unpack_A_'s context protocol around per-tile (Z = 4 * ww, (b' row, zero A) x 2).
inline void p2_seed_unpack_window(uint32_t cnt, uint32_t ww, uint32_t wrap_at, uint32_t reset_to) {
    const uint32_t id = get_operand_id(cb_bprime);
    const uint32_t address = get_local_cb_interface(id).fifo_rd_ptr - 1;
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
#endif
#ifdef TRISC_MATH
// Seed math of DEST tiles 0..cnt-1: bank pair 1 (b' face-0 row) writes DEST rows 0-15 and 32-47, bank pair 2
// (face-1 row) rows 16-31 and 48-63; srcA (zeroed) is re-read from row 0 for each face.
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
#endif
// Once per chunk: formats (srcA bf16 x, srcB Float32 a / b'), the ELWMUL ROW HiFi3 math init (its ADDR_MOD_0 also
// drives the seed) and the AB ROW unpack init with B = one row per face (its x-end also serves the seed).
ALWI void p2_init(uint32_t x_cb) {
    reconfig_data_format(x_cb, cb_bprime);
    state_configure(x_cb, cb_chan_scale, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MATH_FIDELITY>(
        x_cb, cb_chan_scale, 0)));
    UNPACK((_llk_unpack_AB_init_<BroadcastType::ROW>(
        ckernel::DEFAULT_TENSOR_SHAPE, P2_ROWB_SHAPE, ckernel::Transpose::None)));
}
// DEST[0..cnt-1] = 0 - b'[ww...] (row broadcast), b' index advancing ww -> wrap_at -> reset_to.
ALWI void p2_seed_window(uint32_t cnt, uint32_t ww, uint32_t wrap_at, uint32_t reset_to) {
    UNPACK((p2_seed_unpack_window(cnt, ww, wrap_at, reset_to)));
    MATH((p2_seed_math_window(cnt)));
}
// DEST[dst] += x[xi] * a[w] (a as a row broadcast), HiFi3.
ALWI void p2_mul_tile(uint32_t x_cb, uint32_t xi, uint32_t w, uint32_t dst) {
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(x_cb, cb_chan_scale, xi, w)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(x_cb, cb_chan_scale, dst, true)));
}

#ifdef TRISC_PACK
// Raw CB state on the packer (perf_experiments/dram_dynamic_balance; no helper offers a producer-side "is there room"
// probe or a consumer choice): free pages = capacity - (pushed - acked); the sequence word is stored straight into
// the reserved page (CB addresses are 16 B units on TRISC).
inline uint32_t pack_free_pages(uint32_t cb) {
    volatile tt_reg_ptr uint32_t* acked_ptr = get_cb_tiles_acked_ptr(cb);
    const uint16_t received = get_local_cb_interface(cb).tiles_received;
    const uint16_t acked = (uint16_t)reg_read((uint32_t)acked_ptr);
    return (uint16_t)(get_local_cb_interface(cb).fifo_num_pages - (uint16_t)(received - acked));
}
inline void pack_seq_word(uint32_t cb, uint32_t w) {
    cb_reserve_back(cb, 1);
    *reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4) = w;
    cb_push_back(cb, 1);
}
#endif

// The output CB of pass-2 chunk q (DRAM interleaved: the first with a free chunk slot, writer's first, announced on
// its sequence CB). Only the packer touches the output CBs, so only it chooses; SHARDED packs into the output shard.
ALWI uint32_t output_cb_for([[maybe_unused]] uint32_t q) {
    uint32_t out_cb = cb_output_tiles;
#ifdef TRISC_PACK
    if constexpr (!SHARDED) {
        for (;;) {
            if (pack_free_pages(cb_output_tiles) >= T) {
                break;
            }
            if (pack_free_pages(cb_output_b) >= T) {
                out_cb = cb_output_b;
                break;
            }
        }
        pack_seq_word(out_cb == cb_output_tiles ? cb_out_seq_a : cb_out_seq_b, q);
    }
#endif
    return out_cb;
}

// DEST = x*a - b' for n tiles of x_cb from tile x0 (x0 a multiple of W: tile i is channel tile i mod W) -> out_cb.
// The caller waits x, a and b'.
ALWI void normalize_tiles(uint32_t x_cb, uint32_t x0, uint32_t n, uint32_t out_cb) {
    p2_init(x_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t w = 0;
    for (uint32_t i = 0; i < n; i += CHAIN_BLOCK) {
        const uint32_t cnt = n - i < CHAIN_BLOCK ? n - i : CHAIN_BLOCK;
        cb_reserve_back(out_cb, cnt);
        tile_regs_acquire();
        p2_seed_window(cnt, w, W, 0);
        for (uint32_t j = 0; j < cnt; ++j) {
            p2_mul_tile(x_cb, x0 + i + j, w, j);
            w = w + 1 == W ? 0 : w + 1;
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) {
            pack_tile(j, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, cnt);
    }
}

ALWI void normalize_chunk(uint32_t rows_valid, uint32_t q) {
    constexpr uint32_t x_cb = cb_input_keep;
    const uint32_t out_cb = output_cb_for(q);
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    const uint32_t n = rows_valid * W;
    normalize_tiles(x_cb, 0, n, out_cb);
    push_pad(out_cb, T - n);
    cb_pop_front(x_cb, T);
}

// Sharded RM (UNTILIZE_OUT): the same normalize, packed straight to the row-major output shard with
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

// rows tile rows of x_cb from ring tile row x_row0 -> out_cb, untilized (rows x W pages). The caller waits x, a, b'.
ALWI void normalize_untilize_rows(uint32_t x_cb, uint32_t x_row0, uint32_t rows, uint32_t out_cb) {
    p2_init(x_cb);
    pack_untilize_dest_init<UB_CT, W>(out_cb);
    for (uint32_t r0 = 0; r0 < rows; r0 += UB_RT) {
        const uint32_t rt = rows - r0 < UB_RT ? rows - r0 : UB_RT;
        cb_reserve_back(out_cb, rt * W);
        for (uint32_t b = 0; b < UB_NB; ++b) {
            const uint32_t c0 = b * UB_CT;
            tile_regs_acquire();
            p2_seed_window(rt * UB_CT, c0, c0 + UB_CT, c0);
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < UB_CT; ++c) {
                    p2_mul_tile(x_cb, (x_row0 + r0 + r) * W + c0 + c, c0 + c, r * UB_CT + c);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<UB_CT, W>(out_cb, rt, b);
            tile_regs_release();
        }
        cb_push_back(out_cb, rt * W);
    }
    pack_untilize_uninit(out_cb);
}

ALWI void normalize_untilize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    normalize_untilize_rows(x_cb, 0, rows_valid, cb_output_rm);
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
        // DEST 2 holds -k (the -1 tile's product): mp - (-k) rounds exactly like k + mp.
        const sfpi::vFloat mean = mp - sfpi::vFloat(sfpi::dst_reg[64 + i]);
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
    // mr = mean * rstd (one fp32 rounding; the stock expand formed it on the FPU from tf32 operands).
    const sfpi::vFloat mean = sfpi::vFloat(sfpi::dst_reg[64 + i]) * rstd;
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

// gn_finalize_sfpu runs inside finalize_raw's DEST window (below): DST 0/1 hold [-S1', -T2] (the -1 tile x the
// gathered sums, rows replicated) and DST 2 holds -k; with -inv_n the scoped finalize leaves [rstd, mean*rstd] in
// DST 0/1 (rows replicated) for the pack.
// RAW SFPU: bypasses the eltwise_chain finalize (CopyTile x3 + 8 full-tile SFPU ops + 2 PackTile, each SFPU op over
// all 32 vectors) - the chain has no vector scope and cannot run inside a matmul's DEST window. Measured (root
// finalize zone, perf_experiments/combine_critical_path): unet_rm_ip_1024x1280 whole op 27.1 -> 24.3 us with the
// publish hand-off (changelog Perf 1); mean*rstd joined it in Perf 2 (perf_experiments/combine_tail "mr").

constexpr uint32_t cb_fin_out = cb_stats_out;  // the root's writer multicasts it into cb_group_stats
constexpr uint32_t N_STATS = 2 * NU + 1;       // cb_group_stats pages: [rstd, mean*rstd] per image, then e
constexpr uint32_t E_IDX = 2 * NU;             // e's page

// RAW LLK (bypasses compute_kernel_lib::matmul_block + a finalize PostComputeFn): the helper acquires DEST
// itself after its input waits, so nothing can be staged in the DEST window ahead of the gathered partials; the k
// tile had to be a fully replicated UnpackToDestFp32 copy issued after the matmul. Here the matmul init and the k
// product (-1 tile x [k row; zero rows] = -k on every row, exact: k is a bf16 value) run before the gather wait, and
// the sum uses the resident -1 tile as in0 (no ones tiles), so -S1' / -T2 / -k feed the finalize with -inv_n.
// One finalize per image of the unit (n_img): its R gathered pairs (image i at pair offset R*i) -> [rstd, mean*rstd]
// in cb_stats_out (2 pages per image, image order).
ALWI void finalize_raw(uint32_t inv_n_bits, uint32_t eps_bits, uint32_t n_img) {
    constexpr uint32_t in0 = cb_neg_one;
    cb_wait_front(cb_group_shift, 1);
    for (uint32_t i = 0; i < n_img; ++i) {
        tile_regs_acquire();
        reconfig_data_format(cb_group_shift, in0);  // srcA = in1 (Float32), srcB = in0 (the -1 tile, bf16)
        matmul_init(in0, cb_group_shift);
        matmul_tiles(in0, cb_group_shift, 0, 0, 2);
        if (i == 0) {
            MaybeDeviceZoneScope("c_wait_gather");
            cb_wait_front(cb_gathered_partials, 2 * R * NU);
        }
        MaybeDeviceZoneScope("c_finalize");
        for (uint32_t k = 0; k < R; ++k) {
            matmul_tiles(in0, cb_gathered_partials, 0, 2 * (R * i + k), 0);
            matmul_tiles(in0, cb_gathered_partials, 0, 2 * (R * i + k) + 1, 1);
        }
        rsqrt_tile_init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            gn_finalize_sfpu, 0, ckernel::VectorMode::None, inv_n_bits ^ 0x80000000u, eps_bits)));
        tile_regs_commit();
        cb_reserve_back(cb_fin_out, 2);
        PACK((pack_reconfig_data_format(cb_fin_out)));
        tile_regs_wait();
        pack_tile(0, cb_fin_out);
        pack_tile(1, cb_fin_out);
        tile_regs_release();
        cb_push_back(cb_fin_out, 2);
    }
    cb_pop_front(cb_gathered_partials, 2 * R * NU);
    cb_pop_front(cb_group_shift, 1);
}

// RAW LLK (bypasses compute_kernel_lib::matmul_block for the two expand products only; measured, expand stage
// unet_rm_ip_1024x1280 2.76 -> 1.29 us, whole op 27.1 -> 25.4 us; changelog Perf 1, Helper bypasses). Mechanism: (1)
// the helper packs every subblock it computes, so a W = 5 / 7 / 10 / 15 row can only be tiled by the largest divisor of
// W that fits DEST (1 / 1 / 2 / 3 tiles): one DEST handshake + pack per 1..3 tiles. Here the last subblock overlaps its
// predecessor and packs only its new tiles (pack_block from a DEST offset), so every subblock is a full 4-tile
// DEST half (W = 5: 2 subblocks instead of 5) with a single short init per product. (2) Under fp32_dest_acc_en
// the helper re-issues pack_reconfig_data_format on every subblock; here once per product.
// Both products read in0 from cb_group_stats ([rstd, mean*rstd] per image, then e; waited by the caller): inner tile 0
// at in0_off, inner tile 1 (the beta map's) at E_IDX. They share one init (same CBs, formats, subblock; rt_dim = 1,
// so kt_dim only has to match the init).
template <uint32_t K, bool INIT>
ALWI void expand_matmul(uint32_t in0_cb, uint32_t in0_off, uint32_t in1_cb, uint32_t out_cb) {
    if constexpr (INIT) {
        reconfig_data_format(in1_cb, in0_cb);
        PACK((pack_reconfig_data_format(out_cb)));
        ckernel::matmul_block_init(in0_cb, in1_cb, EXP_TRANSPOSE, EXP_SBW_E, 1, K_B);
    }
    for (uint32_t sb = 0; sb < EXP_NSB; ++sb) {
        const bool last = sb == EXP_NSB - 1;
        const uint32_t n0 = last ? W - EXP_SBW_E : sb * EXP_SBW_E;
        const uint32_t n_out = last ? EXP_LAST : EXP_SBW_E;
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            ckernel::matmul_block(
                in0_cb, in1_cb, k == 0 ? in0_off : E_IDX, k * W + n0, 0, EXP_TRANSPOSE, EXP_SBW_E, 1, K_B);
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
    const uint32_t n_img = get_arg_val<uint32_t>(arg++);     // images of the unit (root: statistic sets to finalize)
    const uint32_t seg_img0 = get_arg_val<uint32_t>(arg++);  // unit image of this core's first segment
    const uint32_t w_valid = get_arg_val<uint32_t>(arg++);   // TILE: channel tiles holding data
    const uint32_t vc_arg = get_arg_val<uint32_t>(arg++);    // sharded: valid channels of the slice
    const uint32_t n_ops = get_arg_val<uint32_t>(arg++);
    const uint32_t ops_arg = arg;
    const uint32_t vc = SHARDED ? vc_arg : P;

    CircularBuffer chan_scale_cb(cb_chan_scale);
    CircularBuffer chan_work_cb(cb_chan_work);
    CircularBuffer map_cb(cb_group_map);
    CircularBuffer local_partials_cb(cb_local_partials);

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_chan_scale, cb_group_map, cb_local_partials);

    // The constant -1 tile (bf16, in0 of the root finalize's -k and sender-sum matmuls), resident for the whole
    // kernel. Only the unit root reads it.
    if constexpr (IS_ROOT) {
        DataflowBuffer neg_one_dfb(cb_neg_one);
        eltwise_chain(
            IterationShape::one_tile(), FillScalar<Dst::D0>{-1.0f}, PackTile<output(cb_neg_one)>{neg_one_dfb});
    }

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    auto rows_valid_of = [&](uint32_t j) {
        const uint32_t rows_left = rows_core - j * CHUNK_HT;
        return rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
    };

    if constexpr (SHARDED && OPS) {
        // ================= Sharded, fully resident: one block, the core's shard as op words =================
        // Pieces in shard row order: each pushes its tile rows to the input ring (TILE: the shard itself, credited by
        // the reader); OP_SEG_END closes an image segment, which gets its own statistic set (partials, a / b').
        uint32_t ring_tiles = num_chunks * T;  // TILE: the reader's credits (the whole valid shard)
        {
            MaybeDeviceZoneScope("c_pass1");
            if constexpr (IS_RM) {
                uint32_t ring_rows = 0;
                for (uint32_t i = 0; i < n_ops; ++i) {
                    const uint32_t op = get_arg_val<uint32_t>(ops_arg + i);
                    if (op & OP_SKIP) {
                        // A staged position the shard FIFO steps over (its rows come from cb_rm_stage_in).
                        cb_wait_front(cb_input_rm, W);
                        cb_pop_front(cb_input_rm, W);
                        continue;
                    }
                    tilize_piece(op & OP_ROWS, (op & OP_STAGED) != 0);
                    ring_rows += op & OP_ROWS;
                }
                ring_tiles = ring_rows * W;
            }
            {
                MaybeDeviceZoneScope("c_wait_pilot");
                cb_wait_front(cb_pilot_rows, W);
            }
            cb_wait_front(cb_input_tiles, ring_tiles);
            bool map_ready = false;
            uint32_t row0 = 0, seg_rows = 0;
            for (uint32_t i = 0; i < n_ops; ++i) {
                const uint32_t op = get_arg_val<uint32_t>(ops_arg + i);
                if (op & OP_SKIP) {
                    continue;
                }
                seg_rows += op & OP_ROWS;
                if (op & OP_SEG_END) {
                    moments_rows(row0, seg_rows, true, true, w_valid, vc);
                    if (!map_ready) {
                        // Resident for the whole kernel (matmul in1, NoWaitNoPop); the reader builds it in pass 1's
                        // shadow.
                        MaybeDeviceZoneScope("c_wait_map");
                        cb_wait_front(cb_group_map, W);
                        map_ready = true;
                    }
                    MaybeDeviceZoneScope("c_collapse");
                    collapse_with_map(chan_scale_cb, map_cb, local_partials_cb);
                    collapse_with_map(chan_work_cb, map_cb, local_partials_cb);
                    row0 += seg_rows;
                    seg_rows = 0;
                }
            }
            cb_pop_front(cb_input_tiles, ring_tiles);
            cb_pop_front(cb_pilot_rows, W);
        }
        if constexpr (IS_ROOT) {
            finalize_raw(inv_n_bits, eps_bits, n_img);
        }
        {
            MaybeDeviceZoneScope("c_wait_stats");
            cb_wait_front(cb_group_stats, N_STATS);
        }
        if constexpr (HAS_GAMMA) {
            MaybeDeviceZoneScope("c_wait_affine");
            cb_wait_front(cb_gamma_rows, K_B * W);
        }
        // ---------------- pass 2: per segment, expand its image's a / b', normalize its pieces ----------------
        MaybeDeviceZoneScope("c_pass2");
        constexpr uint32_t x_cb = cb_input_keep;
        cb_wait_front(x_cb, ring_tiles);
        uint32_t img = seg_img0, xrow = 0;
        bool expanded = false;
        for (uint32_t i = 0; i < n_ops; ++i) {
            const uint32_t op = get_arg_val<uint32_t>(ops_arg + i);
            if (op & OP_SKIP) {
                if constexpr (IS_RM) {
                    push_pad(cb_output_rm, W);  // the output shard FIFO steps over the staged position
                }
                continue;
            }
            if (!expanded) {
                MaybeDeviceZoneScope("c_expand");
                expand_matmul<1, true>(cb_group_stats, 2 * img, cb_amap, cb_chan_scale);
                expand_matmul<K_B, false>(cb_group_stats, 2 * img + 1, cb_amap, cb_chan_work);
                cb_wait_front(cb_chan_scale, W);
                cb_wait_front(cb_bprime, W);
                expanded = true;
            }
            const uint32_t rows = op & OP_ROWS;
            if constexpr (IS_RM) {
                normalize_untilize_rows(x_cb, xrow, rows, (op & OP_STAGED) ? cb_rm_stage_out : cb_output_rm);
            } else {
                normalize_tiles(x_cb, xrow * W, rows * W, cb_output_tiles);
            }
            xrow += rows;
            if (op & OP_SEG_END) {
                cb_pop_front(cb_chan_scale, W);
                cb_pop_front(cb_bprime, W);
                ++img;
                expanded = false;
            }
        }
        if constexpr (!IS_RM) {
            push_pad(cb_output_tiles, ring_tiles - xrow * W);  // the writer drains the reader's credit count
        }
        cb_pop_front(x_cb, ring_tiles);
        cb_pop_front(cb_group_stats, N_STATS);
        if constexpr (HAS_GAMMA) {
            cb_pop_front(cb_gamma_rows, K_B * W);
        }
        return;
    }

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
                moments_rows(0, rows_core, true, true, W, vc);
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
                    moments_rows(0, rows_valid, j == 0, j + 1 == num_chunks, W, vc);
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
            finalize_raw(inv_n_bits, eps_bits, 1);
        }

        // ---------------- expand_stats_to_channels ----------------
        {
            // The combine round trip seen from this core: [rstd, mean] multicast back from the unit root.
            MaybeDeviceZoneScope("c_wait_stats");
            cb_wait_front(cb_group_stats, N_STATS);
        }
        if constexpr (HAS_GAMMA) {
            // The reader builds the gamma / beta maps in the combine's stall shadow.
            MaybeDeviceZoneScope("c_wait_affine");
            cb_wait_front(cb_gamma_rows, K_B * W);
        }
        {
            MaybeDeviceZoneScope("c_expand");
            // a = rstd x amap; b' = [mr, e] x [amap; bmap] (in0 tiles 1..K_B of cb_group_stats).
            expand_matmul<1, true>(cb_group_stats, 0, cb_amap, cb_chan_scale);
            expand_matmul<K_B, false>(cb_group_stats, 1, cb_amap, cb_chan_work);
            cb_pop_front(cb_group_stats, N_STATS);
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
                normalize_chunk(rows_valid, jj);
            }
            if (reread) {
                skip_chunk(cb_input_tiles);  // the pass-1 view also carried the re-read
            }
        }
#ifdef TRISC_PACK
        if constexpr (!SHARDED) {
            pack_seq_word(cb_out_seq_a, OUT_SEQ_END);
            pack_seq_word(cb_out_seq_b, OUT_SEQ_END);
        }
#endif
        cb_pop_front(cb_chan_scale, W);
        cb_pop_front(cb_bprime, W);
    }
}
