// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm compute (op_design.md -> Block schedule). The core's rows run in NP pass blocks of PW row waves
// of Q_W tile-rows (Phase 0: one pass, the whole assignment resident); waves are numbered across passes, so the
// exchange is unchanged. Each pass is Phase A + Phase B below. L1 fallback ladder (Refinement 1, host
// blocking_options): GAMMA_MODE 0 x·γ of the pass block / 1 per wave / 2 per C-chunk (GAMMA_STREAM: γ chunks streamed
// by the writer), passes (PW < NW), SQ_ROWS tile-rows per square -> collapse round.
//
// Phase A (needs no remote data), per wave w:
//   square_collapse_wave   eltwise_chain x*x DEST-accumulated per tile-row -> cb_sumsq_partial, then
//                          reduce<SUM, REDUCE_ROW> -> cb_partial_col (D3: the slice partial for the owner) or
//                          cb_chip_partial (D1: the chip partial itself). Sharded W-pad: the base's partial-scaler
//                          mask on the core holding tile Wt-1.
//   chip_sum_wave (owner)  reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip> over the G_W gathered partials ->
//                          cb_chip_partial — COMBINE_LAG waves behind the square (0 since Refinement 2: the reader
//                          contributes / publishes wave w right after it landed; 1 = Phase 0, the gather of wave w
//                          landing while wave w+1 is squared).
// scale_by_gamma_block     x * γ (Row broadcast) for the whole block -> cb_x_gamma (the stall shadow of the exchange)
// Phase B, per wave w:
//   cross_device_finalize  (owner) fold the G devices' records of each owned row in group-position order — SFPU fp32
//                          adds over the record vectors of G DEST slots — then rsqrt(ss/W + eps), one SFPU pass, ->
//                          the 128 B inv record (rows 0-1 of face 0) of a cb_rms_root page
//   scale_by_inv_rms_wave  (x*γ | x) * inv_rms (Col broadcast) -> cb_output
//
// Raw-LLK uses (helper mismatch, op_design.md -> "Helpers considered and rejected"):
//   cross-device fold + finalize (perf_experiments/fold_finalize; fold-start -> all-gather-start on the critical owner
//     1.07 -> 0.25 us, isolated bench 1030 -> 251 ns per G=4 row, bit-identical to the copy_tile + add_binary_tile
//     fold it replaces). Records land compact: [row][position] cb_stats pages, the 128 B record contiguous at the
//     page start (rows 0-1 of face 0, one NoC read). Bypassed and why (capability gaps, not ergonomics):
//     - copy_tile / copy_block (UnpackToDestFp32): one math<->unpack handshake per tile (~80 ns each, the dominant
//       cost) and the whole 4 KiB tile; no API unpacks N tiles to DEST in one transaction or a sub-face row range.
//       ff_unpack_records_to_dest: _llk_unpack_A_ with the MOP replaced by one 32-datum UNPACR per record + a z hop.
//     - add_binary_tile: fixed VectorMode::RC / ITERATIONS=8 (32 vector ops per add, 2 carry data) and one SFPU call
//       per position; no helper folds G DEST slots into one. ff_fold_body: one pass, G loads + G-1 fp32 adds (same
//       SFPU ops, same order as the per-position add) + the finalize, over vectors 0 / 1 of face 0.
//     - pack_tile: packs the CB's full tile shape; the 128 B record is packed with the face_r_dim 2 / 1-face pack MOP
//       (llk_pack_init_impl), restored to the full tile after the push.
//     reduce<AccumulateViaAdd> was never an option (adds through SrcA/SrcB in tf32; the partials must sum in fp32).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Measurement-only stage zones (host DRMS_PERF_ZONES=1 -> define DRMS_PERF_ZONES; off in production).
#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;

namespace {

#ifdef TRISC_MATH
// Base op finalize_col0_body: inv_rms = rsqrt(ss * (1/W) + eps) over the even column-0 vectors of faces 0/2.
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}

// The fused fold + finalize: one SFPU pass over the record vectors of G DEST slots (position order), then
// inv = rsqrt(ss * (1/W) + eps) on the sum. A zero inv (Σx² overflowed to +inf) is written as -0: the all-gathered
// record slot uses +0 as "not arrived yet", and -0 · x scales exactly like +0.
// acc + slot P + ... + slot N-1. bf16 DEST: every partial sum round-trips DEST, like the per-position
// add_binary_tile (its result is written to DEST before the next add reads it) -> bit-identical in both DEST modes.
template <uint32_t P, uint32_t N>
sfpi_inline sfpi::vFloat ff_add_slots(sfpi::vFloat acc) {
    if constexpr (P >= N) {
        return acc;
    } else {
        acc = acc + sfpi::dst_reg[P * 32];
        if constexpr (!DST_ACCUM_MODE) {
            sfpi::dst_reg[0] = acc;
            acc = sfpi::dst_reg[0];
        }
        return ff_add_slots<P + 1, N>(acc);
    }
}
// vectors 0 / 1 of the face (VectorMode::None: face 0): slot 0 += slots 1..N-1, then (FIN) the finalize
template <uint32_t N, bool FIN>
sfpi_inline void ff_fold_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat acc = ff_add_slots<1, N>(sfpi::dst_reg[0]);
        if constexpr (FIN) {
            const sfpi::vFloat mean_sq =
                acc * ckernel::sfpu::Converter::as_float(inv_w_bits) + ckernel::sfpu::Converter::as_float(eps_bits);
            sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
            if constexpr (!DST_ACCUM_MODE) {
                t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
            }
            v_if(t == 0.0f) { t = sfpi::setsgn(t, 1); }
            v_endif;
            sfpi::dst_reg[0] = t;
        } else {
            sfpi::dst_reg[0] = acc;
        }
        sfpi::dst_reg++;
    }
}
#endif

#ifdef TRISC_UNPACK
// N records (consecutive cb_stats pages from first_tile) -> DEST slots s0..s0+N-1 (s0 = the math side's datacopy
// index) in ONE unpack-to-dest transaction: _llk_unpack_A_ with the MOP replaced by FACES UNPACRs per record (both
// address channels z += 1 per face; x_end set by the init) and a z hop to the next page / slot. One math<->unpack
// handshake per N records instead of one per record.
template <uint32_t N, uint32_t FACES>
inline void ff_unpack_records_to_dest(const uint32_t operand, const uint32_t first_tile) {
    const uint32_t id = get_operand_id(operand);
    const uint32_t address =
        get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * first_tile;
    const uint32_t dst_format = unpack_dst_format[id];
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[(unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    set_dst_write_addr(unp_cfg_context, dst_format);
    wait_for_dest_available();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#pragma GCC unroll 8
    for (uint32_t p = 0; p < N; ++p) {
#pragma GCC unroll 2
        for (uint32_t f = 0; f < FACES; ++f) {
            TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        if (p + 1 < N) {
            TTI_INCADCZW(p_setadc::UNP_A, 0, 4 - FACES, 0, 4 - FACES);
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context, dst_format);
    switch_config_context(unp_cfg_context);
}
#endif

constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq_partial = 3;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_x_gamma = 7;
constexpr uint32_t cb_x_reread = 8;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_sumsq_last = 12;
constexpr uint32_t cb_sumsq_acc = 13;
constexpr uint32_t cb_chip_partial = 14;
constexpr uint32_t cb_stats = 15;
constexpr uint32_t cb_output = 16;

// cross_device_finalize records: [row][position] cb_stats pages, each 128 B record contiguous at its page start (rows
// 0-1 of face 0). The G records of a row fold in chunks of the DEST capacity (half sync: 4 fp32 / 8 bf16 slots): the
// first chunk into slots 0.., later ones into slots 1.. on top of the running sum in slot 0 (((r0 + r1) + r2) + ...).
// fp32 DEST: one unpack-to-dest transaction per chunk (one UNPACR of 32 datums per record); bf16 DEST: copy_tile.
constexpr uint32_t FF_DEST_CAP = DST_ACCUM_MODE ? 4 : 8;
template <uint32_t G, uint32_t Q, uint32_t DONE>
ALWI void ff_fold_chunks(uint32_t r, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (DONE < G) {
        constexpr bool first = DONE == 0;
        constexpr uint32_t room = first ? FF_DEST_CAP : FF_DEST_CAP - 1;
        constexpr uint32_t n = (G - DONE) < room ? (G - DONE) : room;
        constexpr uint32_t s0 = first ? 0 : 1;
        if constexpr (DST_ACCUM_MODE) {
            UNPACK((ff_unpack_records_to_dest<n, 1>(cb_stats, r * G + DONE)));
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    s0, cb_stats)));
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_stats, r * G + DONE + i, s0 + i);
            }
        }
        // vectors 0 / 1 of face 0 (rows 0-3, even / odd columns) hold the record's 32 values (rows 0-1)
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ff_fold_body < n + s0, DONE + n == G >, 0, ckernel::VectorMode::None, inv_w_bits, eps_bits)));
        ff_fold_chunks<G, Q, DONE + n>(r, inv_w_bits, eps_bits);
    }
}
ALWI void ff_init() {
    copy_init(cb_stats);  // math: the plain (NONE) datacopy; bf16 DEST: the whole-tile copy
    if constexpr (DST_ACCUM_MODE) {
        // the unpack-to-dest MOP state, x_end = 32 datums per UNPACR (rows 0-1 of a face)
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false, false, cb_stats)));
        UNPACK((config_unpacker_x_end<p_setadc::UNP_A>(2)));
    }
    // rows 0-1 of face 0 (face_r_dim 2, 1 face): the finalized 128 B (fp32) / 64 B (bf16) inv record at the page start
    PACK((llk_pack_init_impl<PackMode::Default, false, false, false>(
        pack_src_format[get_output_id(cb_rms_root)], 2, 32, 1, 1, false)));
}
}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr bool SHARDED = get_compile_time_arg_val(1) != 0;
    constexpr bool COMBINE = get_compile_time_arg_val(2) != 0;
    constexpr bool MASKED_CFG = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t WS = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t NCH = get_compile_time_arg_val(6);
    constexpr uint32_t NW = get_compile_time_arg_val(7);
    constexpr uint32_t Q_W = get_compile_time_arg_val(8);
    constexpr uint32_t OWNERS = get_compile_time_arg_val(9);
    constexpr uint32_t Q = get_compile_time_arg_val(10);
    constexpr uint32_t G = get_compile_time_arg_val(11);
    constexpr uint32_t G_W = get_compile_time_arg_val(12);
    constexpr uint32_t SCALE_BLOCK = get_compile_time_arg_val(13);
    // output placement: shard-backed cb_output (packed in place, strided; the input's own placement) or streamed to
    // the writer per tile-row (interleaved output, or a resharded one)
    constexpr bool OUT_SHARDED = get_compile_time_arg_val(14) != 0;
    // γ placement of the scaling pass (L1 fallback ladder): 0 the whole pass block, 1 per wave, 2 per C-chunk
    constexpr uint32_t GAMMA_MODE = get_compile_time_arg_val(15);
    constexpr bool GAMMA_PER_WAVE = GAMMA_MODE == 1;
    constexpr bool GAMMA_PER_CHUNK = GAMMA_MODE == 2;
    // waves per pass: the core's rows run in ceil(NW / PW) pass blocks (Phase A + Phase B each); NW == PW is Phase 0
    constexpr uint32_t PW = get_compile_time_arg_val(16);
    constexpr uint32_t NP = (NW + PW - 1) / PW;
    // cb_gamma_tiles streams one C-tile chunk per (wave, chunk) from the writer instead of the resident slice
    constexpr bool GAMMA_STREAM = get_compile_time_arg_val(17) != 0;
    // tile-rows per square -> collapse round (cb_sumsq_partial / cb_sumsq_last / cb_sumsq_acc capacity)
    constexpr uint32_t SQ_ROWS = get_compile_time_arg_val(18);
    // x re-read (L1 fallback, interleaved only): Phase A squares x streamed per round of SQ_ROWS tile-rows (popped
    // per round, never resident); Phase B scales x re-read by the writer one (wave, chunk, tile-row) item at a time
    // into cb_x_reread — x*γ per row into cb_x_gamma (C tiles), then * inv_rms -> cb_output
    constexpr bool REREAD = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t WP = NCH * C;
    // waves between a wave's square and its chip sum (the reader's service lag, the same CT value): 1 / 0
    constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(20);
    // waves an owner's cross-device fold may run ahead of its own scaling pass (>= NW: every owned fold first)
    constexpr uint32_t FOLD_AHEAD = get_compile_time_arg_val(21);

    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t V = get_arg_val<uint32_t>(1);
    const uint32_t my_j = get_arg_val<uint32_t>(2);
    const bool masked_core = get_arg_val<uint32_t>(3) != 0;
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);

    compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial);

    constexpr uint32_t cb_collapsed = COMBINE ? cb_partial_col : cb_chip_partial;

    auto n_wave = [&](uint32_t w) -> uint32_t {
        const uint32_t r0 = w * Q_W;
        return r0 >= rows ? 0 : ((rows - r0) < Q_W ? rows - r0 : Q_W);
    };
    // my owner slot k in wave w (owner k of wave w = member (w*OWNERS + k) mod G_W), or OWNERS if none
    auto my_k = [&](uint32_t w) -> uint32_t {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            if ((w * OWNERS + k) % G_W == my_j) {
                return k;
            }
        }
        return OWNERS;
    };
    auto n_own = [&](uint32_t w, uint32_t k) -> uint32_t {
        const uint32_t n = n_wave(w);
        const uint32_t first = k * Q;
        return first >= n ? 0 : ((n - first) < Q ? n - first : Q);
    };

    using SumsqPack = ckl::PackTile<ckl::output(
        cb_sumsq_partial,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow)>;
    constexpr auto x_strided_input = ckl::input(
        cb_x_tiles,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    using SquareAccumulateStrided = ckl::
        BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, x_strided_input, ckl::Dst::D0, ckl::DestAccumulation::PerRow>;
    constexpr auto x_gamma_strided_input = ckl::input(
        cb_x_gamma,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto x_reread_input = ckl::input(
        cb_x_reread,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto inv_rms_strided_input = ckl::input(
        cb_inv_rms,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto gamma_row_input = ckl::input(
        cb_gamma_tiles,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
    constexpr auto row_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter);
    };
    constexpr auto shard_output = ckl::output(
        cb_output,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);

    auto collapse = [&](uint32_t n, auto partial_scaler, auto accumulate, auto cb_in_tag, auto cb_out_tag) {
        constexpr uint32_t cb_in = decltype(cb_in_tag)::value;
        constexpr uint32_t cb_out = decltype(cb_out_tag)::value;
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_in,
            cb_reduce_scaler,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::of(n, 1),
            ckl::ReduceInputMemoryLayout::contiguous(),
            accumulate,
            ckl::NoOp{},
            partial_scaler);
    };
    using CbSumsqPartial = std::integral_constant<uint32_t, cb_sumsq_partial>;
    using CbSumsqLast = std::integral_constant<uint32_t, cb_sumsq_last>;
    using CbSumsqAcc = std::integral_constant<uint32_t, cb_sumsq_acc>;
    using CbCollapsed = std::integral_constant<uint32_t, cb_collapsed>;

    const uint32_t full_tiles = masked_core ? V - 1 : V;

    // x CB row of core row r: the resident shard holds every row; an interleaved pass block starts at its pass row
    uint32_t x_row0 = 0;
    auto xrow = [&](uint32_t r) -> uint32_t { return SHARDED ? r : r - x_row0; };

    // square + collapse of n consecutive tile-rows starting at x tile xbase
    auto square_collapse_rows = [&](uint32_t xbase, uint32_t n) {
        if (full_tiles > 0) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, full_tiles),
                SquareAccumulateStrided{ckl::StridedTileRange{xbase, WS}, ckl::StridedTileRange{xbase, WS}},
                SumsqPack{});
        }
        if constexpr (MASKED_CFG) {
            if (masked_core) {
                const ckl::StridedTileRange last_tile{xbase + V - 1, WS};
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(n, 1),
                    SquareAccumulateStrided{last_tile, last_tile},
                    ckl::PackTile<ckl::output(
                        cb_sumsq_last,
                        ckl::ReservePolicy::PerOuter,
                        ckl::PushPolicy::PerOuter,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Direct,
                        ckl::DestAccumulation::PerRow)>{});
                if (full_tiles > 0) {
                    collapse(
                        n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbSumsqAcc{});
                    collapse(
                        n,
                        ckl::ReducePartialScaler::with_partial(),
                        ckl::Accumulate::at(cb_sumsq_acc, 1),
                        CbSumsqLast{},
                        CbCollapsed{});
                } else {
                    collapse(
                        n,
                        ckl::ReducePartialScaler::with_partial(),
                        ckl::NoAccumulation{},
                        CbSumsqLast{},
                        CbCollapsed{});
                }
                return;
            }
        }
        collapse(n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbCollapsed{});
    };

    // ---- square_collapse_wave ----
    auto square_collapse = [&](uint32_t w) {
        const uint32_t n = n_wave(w);
        if (n == 0) {
            return;
        }
        if constexpr (REREAD) {
            // streamed rounds: each round's rows sit at the CB front, popped once collapsed
            DRMS_ZONE("cmp_square");
            for (uint32_t i = 0; i < n; i += SQ_ROWS) {
                const uint32_t m = (n - i) < SQ_ROWS ? n - i : SQ_ROWS;
                cb_wait_front(cb_x_tiles, m * WS);
                square_collapse_rows(0, m);
                cb_pop_front(cb_x_tiles, m * WS);
            }
            return;
        }
        const uint32_t r0 = xrow(w * Q_W);
        const uint32_t xbase = r0 * WS;
        if constexpr (!SHARDED) {
            DRMS_ZONE("cmp_wait_x");
            cb_wait_front(cb_x_tiles, (r0 + n) * WS);
        }
        DRMS_ZONE("cmp_square");
        // SQ_ROWS rows per square -> collapse round (the whole wave by default; fewer under L1 pressure: the raw Σx²
        // CBs then hold SQ_ROWS tiles, the collapsed rows still accumulate in the wave's cb_partial_col)
        for (uint32_t i = 0; i < n; i += SQ_ROWS) {
            const uint32_t m = (n - i) < SQ_ROWS ? n - i : SQ_ROWS;
            square_collapse_rows(xbase + i * WS, m);
        }
    };

    // ---- chip_sum_wave (owner): Σ over the G_W slice partials of the owned rows ----
    auto chip_sum = [&](uint32_t w) {
        if constexpr (COMBINE) {
            const uint32_t k = my_k(w);
            if (k == OWNERS) {
                return;
            }
            const uint32_t no = n_own(w, k);
            if (no == 0) {
                return;
            }
            DRMS_ZONE("cmp_chipsum");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_gather,
                cb_reduce_scaler,
                cb_chip_partial,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::AccumulateViaAdd,
                ckl::ReduceWithinTile::Skip>(
                ckl::ReduceInputBlockShape::of(G_W, no),
                ckl::ReduceInputMemoryLayout::with_row_stride(Q),
                ckl::NoAccumulation{},
                ckl::NoOp{});
        }
    };

    // ---- cross_device_finalize (owner): position-order fp32 fold of the G records + column-0 finalize ----
    auto fold_finalize = [&](uint32_t no) {
        // every init before the records' wait: the unpack thread runs them in the exchange stall
        reconfig_data_format_srca(cb_stats);
        pack_reconfig_data_format(cb_rms_root);
        ff_init();
        add_binary_tile_init();
        rsqrt_tile_init();  // the sqrt body's programmable constants
        cb_reserve_back(cb_rms_root, no);
        {
            DRMS_ZONE("cmp_wait_stats");
            cb_wait_front(cb_stats, G * Q);
        }
        {
            DRMS_ZONE("cmp_fold");
            for (uint32_t r = 0; r < no; ++r) {
                tile_regs_acquire();
                ff_fold_chunks<G, Q, 0>(r, inv_w_bits, eps_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rms_root);
                tile_regs_release();
            }
            cb_push_back(cb_rms_root, no);
        }
        PACK((llk_pack_init(cb_rms_root)));  // the full-tile pack MOP back (after the push: off the chain)
        cb_pop_front(cb_stats, G * Q);
    };

    if constexpr (SHARDED) {
        cb_wait_front(cb_x_tiles, rows * WS);
    }

    // ================= scale_by_gamma =================
    // x * γ for x CB tile-rows [xr0, xr0 + n) -> cb_x_gamma (WS tiles pushed per row).
    auto gamma_pass = [&](uint32_t xr0, uint32_t n) {
        if constexpr (HAS_GAMMA) {
            {
                DRMS_ZONE("cmp_wait_gamma");
                cb_wait_front(cb_gamma_tiles, WP);
            }
            DRMS_ZONE("cmp_gamma");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, WS).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, gamma_row_input>{
                    ckl::StridedTileRange{xr0 * WS, WS}, 0u},
                ckl::PackTile<row_output(cb_x_gamma)>{});
        }
    };

    // (x*γ | x) * inv_rms (Col broadcast) of n rows, chunk columns [c0, c0 + cw): src = (CB tile offset, row stride)
    auto scale_chunk = [&](auto src_input_tag,
                           ckl::StridedTileRange src,
                           uint32_t n,
                           uint32_t cw,
                           uint32_t inv_r0,
                           uint32_t out_tile0) {
        constexpr auto src_input = decltype(src_input_tag)::value;
        const ckl::StridedTileRange inv_range{inv_r0, 1};
        if constexpr (OUT_SHARDED) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, src_input, inv_rms_strided_input>{src, inv_range},
                ckl::PackTile<shard_output>{ckl::StridedTileRange{out_tile0, WS}});
        } else {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, src_input, inv_rms_strided_input>{src, inv_range},
                ckl::PackTile<row_output(cb_output)>{});
        }
    };
    using XIn = std::integral_constant<decltype(x_strided_input), x_strided_input>;
    using XGammaIn = std::integral_constant<decltype(x_gamma_strided_input), x_gamma_strided_input>;
    using XRereadIn = std::integral_constant<decltype(x_reread_input), x_reread_input>;

    if constexpr (OUT_SHARDED) {
        cb_reserve_back(cb_output, rows * WS);
    }

    for (uint32_t p = 0; p < NP; ++p) {
        const uint32_t w_begin = p * PW;
        const uint32_t w_end = (w_begin + PW < NW) ? w_begin + PW : NW;
        const uint32_t row0 = w_begin * Q_W;
        const uint32_t row_end = (w_end * Q_W < rows) ? w_end * Q_W : rows;
        const uint32_t rows_p = row_end > row0 ? row_end - row0 : 0;
        x_row0 = row0;

        // ================= Phase A =================
        for (uint32_t w = w_begin; w < w_end + COMBINE_LAG; ++w) {
            if (w < w_end) {
                square_collapse(w);
            }
            if (w >= w_begin + COMBINE_LAG) {
                chip_sum(w - COMBINE_LAG);
            }
        }

        // Whole pass block (default): the γ pass of every wave is the stall shadow of the exchange. Per wave / per
        // chunk (L1 fallbacks): it runs right before the wave's (chunk's) inv scaling.
        if constexpr (HAS_GAMMA && GAMMA_MODE == 0 && !REREAD) {
            if (rows_p > 0) {
                gamma_pass(xrow(row0), rows_p);
                cb_wait_front(cb_x_gamma, rows_p * WS);
            }
        }

        // ================= Phase B =================
        // cross_device_finalize of the owned waves up to FOLD_AHEAD waves ahead of the scaling (in wave order). Large
        // FOLD_AHEAD: an owner's fold of wave w+1 never waits behind its scaling of wave w (another owner's
        // all-gather), but its own earlier waves are scaled / stored only after its last owned records landed. 0: the
        // owner stores its earlier waves as their inv arrives. Deadlock-free for any value: an owner folds wave w
        // before it scales wave w, and scale(w) only waits on folds of waves <= w.
        uint32_t wf = w_begin;  // next wave whose (owned) fold is due
        auto fold_upto = [&](uint32_t lim) {
            for (; wf < w_end && wf <= lim; ++wf) {
                const uint32_t k = my_k(wf);
                if (k < OWNERS) {
                    const uint32_t no = n_own(wf, k);
                    if (no > 0) {
                        fold_finalize(no);
                    }
                }
            }
        };
        for (uint32_t w = w_begin; w < w_end; ++w) {
            fold_upto(w + FOLD_AHEAD);
            const uint32_t n = n_wave(w);
            const uint32_t r0 = w * Q_W;
            const uint32_t xr0 = xrow(r0);
            if constexpr (HAS_GAMMA && GAMMA_PER_WAVE) {
                if (n > 0) {
                    gamma_pass(xr0, n);
                    cb_wait_front(cb_x_gamma, n * WS);
                }
            }
            const uint32_t inv_r0 = (w - w_begin) * Q_W;
            {
                DRMS_ZONE("cmp_wait_inv");
                cb_wait_front(cb_inv_rms, inv_r0 + Q_W);
            }
            DRMS_ZONE("cmp_scale");
            if (n == 0) {
                continue;
            }
            // x*γ rows: block mode addresses the wave inside the pass block; per-wave mode the wave is the front
            [[maybe_unused]] const uint32_t xg0 = GAMMA_PER_WAVE ? 0 : (r0 - row0) * WS;
            if constexpr (REREAD) {
                for (uint32_t c = 0; c < NCH; ++c) {
                    const uint32_t cw = (c + 1 == NCH) ? WS - c * C : C;
                    if constexpr (HAS_GAMMA) {
                        DRMS_ZONE("cmp_wait_gamma");
                        cb_wait_front(cb_gamma_tiles, GAMMA_STREAM ? C : WP);
                    }
                    for (uint32_t r = 0; r < n; ++r) {
                        {
                            DRMS_ZONE("cmp_wait_xr");
                            cb_wait_front(cb_x_reread, cw);
                        }
                        if constexpr (HAS_GAMMA) {
                            ckl::eltwise_chain(
                                ckl::IterationShape::grid(1, cw).block_size(SCALE_BLOCK),
                                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_reread_input, gamma_row_input>{
                                    ckl::StridedTileRange{0, cw}, GAMMA_STREAM ? 0u : c * C},
                                ckl::PackTile<row_output(cb_x_gamma)>{});
                            cb_wait_front(cb_x_gamma, cw);
                            scale_chunk(XGammaIn{}, ckl::StridedTileRange{0, cw}, 1, cw, inv_r0 + r, 0);
                            cb_pop_front(cb_x_gamma, cw);
                        } else {
                            scale_chunk(XRereadIn{}, ckl::StridedTileRange{0, cw}, 1, cw, inv_r0 + r, 0);
                        }
                        cb_pop_front(cb_x_reread, cw);
                    }
                    if constexpr (HAS_GAMMA && GAMMA_STREAM) {
                        cb_pop_front(cb_gamma_tiles, C);
                    }
                }
                continue;
            }
            for (uint32_t c = 0; c < NCH; ++c) {
                const uint32_t cw = (c + 1 == NCH) ? WS - c * C : C;
                const ckl::StridedTileRange x_range{xr0 * WS + c * C, WS};
                if constexpr (HAS_GAMMA && GAMMA_PER_CHUNK) {
                    // per-chunk x*γ (n rows x cw tiles) ahead of its inv scaling; γ resident or streamed per chunk
                    {
                        DRMS_ZONE("cmp_wait_gamma");
                        cb_wait_front(cb_gamma_tiles, GAMMA_STREAM ? C : WP);
                    }
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, gamma_row_input>{
                            x_range, GAMMA_STREAM ? 0u : c * C},
                        ckl::PackTile<row_output(cb_x_gamma)>{});
                    if constexpr (GAMMA_STREAM) {
                        cb_pop_front(cb_gamma_tiles, C);
                    }
                    cb_wait_front(cb_x_gamma, n * cw);
                    scale_chunk(XGammaIn{}, ckl::StridedTileRange{0, cw}, n, cw, inv_r0, r0 * WS + c * C);
                    cb_pop_front(cb_x_gamma, n * cw);
                } else if constexpr (HAS_GAMMA) {
                    scale_chunk(XGammaIn{}, ckl::StridedTileRange{xg0 + c * C, WS}, n, cw, inv_r0, r0 * WS + c * C);
                } else {
                    scale_chunk(XIn{}, x_range, n, cw, inv_r0, r0 * WS + c * C);
                }
            }
            if constexpr (HAS_GAMMA && GAMMA_PER_WAVE) {
                cb_pop_front(cb_x_gamma, n * WS);
            }
        }

        // ================= release_pass =================
        if constexpr (!SHARDED && !REREAD) {
            cb_pop_front(cb_x_tiles, rows_p * WS);
        }
        cb_pop_front(cb_inv_rms, (w_end - w_begin) * Q_W);
        if constexpr (HAS_GAMMA && GAMMA_MODE == 0 && !REREAD) {
            cb_pop_front(cb_x_gamma, rows_p * WS);
        }
    }
    if constexpr (OUT_SHARDED) {
        cb_push_back(cb_output, rows * WS);
    }

    // ================= release_block =================
    if constexpr (SHARDED) {
        cb_pop_front(cb_x_tiles, rows * WS);
    }
    if constexpr (HAS_GAMMA && !GAMMA_STREAM) {
        cb_pop_front(cb_gamma_tiles, WP);
    }
    cb_pop_front(cb_reduce_scaler, MASKED_CFG ? 2 : 1);
}
