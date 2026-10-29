// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fold_finalize micro-bench: the owner's cross-device fold (G fp32 records, data in row 0) + row-0 finalize ->
// cb_rms_root, REPS times, each rep gated by the DM kernel's push of cb_stats (latency, not throughput).
//
// VARIANT
//   0 base        the op's fold_finalize verbatim: copy_tile (UnpackToDestFp32, full tile) x G into 2 DEST slots,
//                 G-1 full-tile add_binary_tile, finalize_row0_body (VectorMode::R), pack_tile.
//   1 hoist       base, the unpack-side reconfig / copy_init issued BEFORE cb_wait_front(cb_stats).
//   2 slots       hoist + copy_tile of the G records into G DEST slots + ONE custom SFPU pass (row-0 vectors only:
//                 load G, add in position order, finalize, store).
//   3 slots_bcast hoist + unary_bcast<ROW> (UnpackToDestFp32 unpacks ONLY row 0 of faces 0/1, math broadcasts it)
//                 into G slots + the custom SFPU pass.
//   4 slots_row   hoist + raw LLK: ROW unpack-to-dest MOP (row 0 of faces 0/1 only) with the plain NONE datacopy on
//                 math (no broadcast MOVs) into G slots + the custom SFPU pass.
//   6 slots_row_pk  slots_row + row-only pack (face_r_dim 1, 2 faces): the joined 128 B record, not a 4 KiB tile.
//   7 multi       one unpack-to-dest transaction for the G records of a row (records land [row][position]),
//                 ROW-face UNPACRs, + the custom SFPU pass.
//   8 multi_pk    multi + the row-only pack.
//   9 compact     records land as 128 B contiguous (rows 0-1 of face 0): 1 UNPACR / record, SFPU on face 0 only
//                 (2 vectors), pack of rows 0-1 of face 0 (the joined 128 B record).
//  10 op_compact  the op copy's DRMS_FF_MODE 2 fold (generic G: DEST-capacity chunks; bf16 DEST: copy_tile).
//   5 radd        base structure (2 slots), adds scoped to row 0 (VectorMode::R, ITERATIONS=2) — option (a).
//
// Raw-LLK / custom-SFPU (measured justification in the bench README-less report):
//   add_binary_tile: fixed VectorMode::RC / ITERATIONS=8 — 32 vectors per add where 4 carry data, and one SFPU call
//     per position; a G-way fold of DEST slots into one has no helper.
//   copy_tile: always unpacks the whole 4 KiB fp32 tile; the ROW unpack-to-dest MOP is only reachable through
//     unary_bcast (which adds the math broadcast MOVs).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

#if defined(FF_ZONES) && FF_ZONES
#define FF_ZONE(name) DeviceZoneScopedN(name)
#else
#define FF_ZONE(name)
#endif

namespace {

#ifdef TRISC_MATH
// The op's finalize_row0_body, verbatim.
sfpi_inline void finalize_row0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        v_if(t == 0.0f) { t = sfpi::setsgn(t, 1); }
        v_endif;
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg++;
    }
}

// acc + slot P + ... + slot N-1 (position order). bf16 DEST: every partial sum is stored to / reloaded from DEST,
// exactly as the per-position add_binary_tile does (its result is written to DEST before the next add reads it).
template <uint32_t P, uint32_t N>
sfpi_inline sfpi::vFloat add_slots(sfpi::vFloat acc) {
    if constexpr (P >= N) {
        return acc;
    } else {
        acc = acc + sfpi::dst_reg[P * 32];
        if constexpr (!DST_ACCUM_MODE) {
            sfpi::dst_reg[0] = acc;
            acc = sfpi::dst_reg[0];
        }
        return add_slots<P + 1, N>(acc);
    }
}

// One pass over the row-0 vectors (0 / 1 of faces 0 / 1 under VectorMode::R): slot 0 += slots 1..N-1, then (FIN)
// the op's finalize on the sum. Same SFPU ops in the same order as base -> bit-identical.
template <uint32_t N, bool FIN>
sfpi_inline void fold_row0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat acc = add_slots<1, N>(sfpi::dst_reg[0]);
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

constexpr uint32_t cb_x_tiles = 1;  // bf16 dummy: the op's hw_startup operand
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_stats = 15;

// DEST slots usable in one acquire (half sync: 4 fp32 / 8 bf16 tiles)
constexpr uint32_t DEST_CAP = DST_ACCUM_MODE ? 4 : 8;

// record tile t -> DEST slot s, by copy method M (2 full copy_tile, 3 unary_bcast<ROW>, 4 raw ROW unpack)
template <uint32_t M>
ALWI void load_record(uint32_t t, uint32_t s) {
    if constexpr (M == 3) {
        unary_bcast<BroadcastType::ROW>(cb_stats, t, s);
    } else if constexpr (M == 4 && DST_ACCUM_MODE) {  // bf16 DEST: not unpack-to-dest -> copy_tile
        UNPACK(
            (llk_unpack_A<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(cb_stats, t)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            s, cb_stats)));
    } else {
        copy_tile(cb_stats, t, s);
    }
}

template <uint32_t M>
ALWI void load_init() {
    if constexpr (M == 3) {
        unary_bcast_init<BroadcastType::ROW>(cb_stats);
    } else if constexpr (M == 4 && DST_ACCUM_MODE) {  // bf16 DEST: not unpack-to-dest -> copy_tile
        copy_init(cb_stats);                          // math: the plain (NONE) unpack-to-dest datacopy
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false, false, cb_stats)));
    } else {
        copy_init(cb_stats);
    }
}

#ifdef TRISC_UNPACK
// N records (consecutive pages from first_tile) -> row 0 of faces 0 / 1 of DEST slots 0..N-1 in ONE unpack-to-dest
// transaction (one math<->unpack handshake instead of N): _llk_unpack_A_ with the ROW MOP replaced by 2 UNPACRs per
// record (ch0 = L1 face, ch1 = DEST face, both z += 1) and a z += 2 hop over faces 2 / 3 to the next record / slot.
// Needs unpacker x_end = 16 datums (the ROW unpack-to-dest init).
template <uint32_t N>
inline void unpack_records_row0_to_dest(const uint32_t operand, const uint32_t first_tile) {
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
        TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        if (p + 1 < N) {
            TTI_INCADCZW(p_setadc::UNP_A, 0, 2, 0, 2);
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context, dst_format);
    switch_config_context(unp_cfg_context);
}
// compact records: each record is 128 B contiguous at its page start (rows 0-1 of face 0) -> rows 0-1 of face 0 of
// DEST slots 0..N-1: ONE UNPACR of 32 datums per record (x_end = 32), z += 3 to the next page / slot.
template <uint32_t N>
inline void unpack_compact_records_to_dest(const uint32_t operand, const uint32_t first_tile) {
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
        TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        if (p + 1 < N) {
            TTI_INCADCZW(p_setadc::UNP_A, 0, 3, 0, 3);
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context, dst_format);
    switch_config_context(unp_cfg_context);
}
#endif

// chunks of the G-way fold: the first loads min(G, CAP) records into slots 0.., the rest up to CAP-1 new records
// into slots 1.. on top of the running sum in slot 0 (position order kept: ((r0 + r1) + r2) + ...)
template <uint32_t M, uint32_t G, uint32_t Q, uint32_t DONE, bool RM = false>
ALWI void fold_chunks(uint32_t r, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (DONE < G) {
        constexpr uint32_t first = DONE == 0;
        constexpr uint32_t room = first ? DEST_CAP : DEST_CAP - 1;
        constexpr uint32_t n = (G - DONE) < room ? (G - DONE) : room;
        for (uint32_t i = 0; i < n; ++i) {
            load_record<M>(RM ? r * G + DONE + i : (DONE + i) * Q + r, i + (first ? 0 : 1));
        }
        constexpr uint32_t slots = first ? n : n + 1;
        constexpr bool fin = DONE + n == G;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            fold_row0_body<slots, fin>, 0, ckernel::VectorMode::R, inv_w_bits, eps_bits)));
        fold_chunks<M, G, Q, DONE + n, RM>(r, inv_w_bits, eps_bits);
    }
}

// all G records of tile-row r (pages r*G .. r*G+G-1, the [row][position] landing) in one transaction + one SFPU pass
template <uint32_t G, uint32_t Q>
ALWI void fold_multi(uint32_t r, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (DST_ACCUM_MODE && G <= DEST_CAP) {
        UNPACK((unpack_records_row0_to_dest<G>(cb_stats, r * G)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, cb_stats)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            fold_row0_body<G, true>, 0, ckernel::VectorMode::R, inv_w_bits, eps_bits)));
    } else {
        fold_chunks<4, G, Q, 0, true>(r, inv_w_bits, eps_bits);
    }
}

template <uint32_t G>
ALWI void fold_compact(uint32_t r, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (DST_ACCUM_MODE && G <= DEST_CAP) {  // bench-only path (variant 10 is the generic one)
        UNPACK((unpack_compact_records_to_dest<G>(cb_stats, r * G)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, cb_stats)));
        // vectors 0 / 1 of face 0 = rows 0-3 even / odd columns: the record's 32 values (rows 0-1) in 2 vector passes
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            fold_row0_body<G, true>, 0, ckernel::VectorMode::None, inv_w_bits, eps_bits)));
    }
}

// ---- variant 10: the op copy's (drms_ff) DRMS_FF_MODE 2 code, verbatim ----
#ifdef TRISC_MATH
// perf_experiments/fold_finalize: the fused row-0 fold + finalize (one SFPU pass over the data vectors of G DEST
// slots, position order, the finalize above on the sum) and the record unpack of DRMS_FF_MODE 1 / 2.
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
// vectors 0 / 1 of the face(s) the VectorMode selects: slot 0 += slots 1..N-1, then (FIN) finalize_row0_body's math
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

// perf_experiments/fold_finalize: DRMS_FF_MODE 1 = records as today ([position][row], row 0 of faces 0 / 1; per-record
// ROW unpack-to-dest, no broadcast) ; 2 = compact records ([row][position], the 128 B record contiguous at the page
// start = rows 0-1 of face 0; one unpack-to-dest transaction per chunk, SFPU on face 0, rows 0-1 of face 0 packed:
// the joined 128 B inv record). G records fold in chunks of the DEST capacity (half sync: 4 fp32 / 8 bf16 slots): the
// first chunk into slots 0.., later ones into slots 1.. on top of the running sum in slot 0 (((r0 + r1) + r2) + ...).
constexpr uint32_t FF_DEST_CAP = DST_ACCUM_MODE ? 4 : 8;
template <uint32_t MODE, uint32_t G, uint32_t Q, uint32_t DONE>
ALWI void ff_fold_chunks(uint32_t r, uint32_t inv_w_bits, uint32_t eps_bits) {
    if constexpr (DONE < G) {
        constexpr bool first = DONE == 0;
        constexpr uint32_t room = first ? FF_DEST_CAP : FF_DEST_CAP - 1;
        constexpr uint32_t n = (G - DONE) < room ? (G - DONE) : room;
        constexpr uint32_t s0 = first ? 0 : 1;
        if constexpr (MODE == 2 && DST_ACCUM_MODE) {
            UNPACK((ff_unpack_records_to_dest<n, 1>(cb_stats, r * G + DONE)));
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    s0, cb_stats)));
        } else if constexpr (MODE == 1 && DST_ACCUM_MODE) {
            for (uint32_t i = 0; i < n; ++i) {
                UNPACK((llk_unpack_A<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
                    cb_stats, (DONE + i) * Q + r)));
                MATH((llk_math_eltwise_unary_datacopy<
                      DataCopyType::A2D,
                      DST_ACCUM_MODE,
                      BroadcastType::NONE,
                      UnpackToDestEn>(s0 + i, cb_stats)));
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_stats, MODE == 2 ? r * G + DONE + i : (DONE + i) * Q + r, s0 + i);
            }
        }
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ff_fold_body < n + s0,
            DONE + n == G >
            , 0, MODE == 2 ? ckernel::VectorMode::None : ckernel::VectorMode::R, inv_w_bits, eps_bits)));
        ff_fold_chunks<MODE, G, Q, DONE + n>(r, inv_w_bits, eps_bits);
    }
}
template <uint32_t MODE>
ALWI void ff_init() {
    copy_init(cb_stats);  // math: the plain (NONE) datacopy; bf16 DEST: the whole-tile copy
    if constexpr (DST_ACCUM_MODE) {
        // the ROW unpack-to-dest MOP + x_end = 16 datums (row 0 of a face); compact: 32 datums (rows 0-1)
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false, false, cb_stats)));
        if constexpr (MODE == 2) {
            UNPACK((config_unpacker_x_end<p_setadc::UNP_A>(2)));
        }
    }
    if constexpr (MODE == 2) {
        // rows 0-1 of face 0 (face_r_dim 2, 1 face): the joined record, 2 x 16 datums at the page start
        PACK((llk_pack_init_impl<PackMode::Default, false, false, false>(
            pack_src_format[get_output_id(cb_rms_root)], 2, 32, 1, 1, false)));
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(2);
    constexpr uint32_t REPS = get_compile_time_arg_val(3);
    constexpr uint32_t NO = get_compile_time_arg_val(4);  // owned rows (<= Q)

    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_x_tiles);
    const uint32_t no = NO;

    for (uint32_t rep = 0; rep < REPS; ++rep) {
        if constexpr (VARIANT == 0) {
            // ---- the op's fold_finalize, verbatim ----
            {
                FF_ZONE("cmp_wait_stats");
                cb_wait_front(cb_stats, G * Q);
            }
            FF_ZONE("cmp_fold");
            reconfig_data_format_srca(cb_stats);
            pack_reconfig_data_format(cb_rms_root);
            cb_reserve_back(cb_rms_root, no);
            copy_init(cb_stats);
            add_binary_tile_init();
            rsqrt_tile_init();
            for (uint32_t r = 0; r < no; ++r) {
                tile_regs_acquire();
                copy_tile(cb_stats, r, 0);
                for (uint32_t pos = 1; pos < G; ++pos) {
                    copy_tile(cb_stats, pos * Q + r, 1);
                    add_binary_tile(0, 1, 0);
                }
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    finalize_row0_body, 0, ckernel::VectorMode::R, inv_w_bits, eps_bits)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rms_root);
                tile_regs_release();
            }
            cb_push_back(cb_rms_root, no);
            cb_pop_front(cb_stats, G * Q);
        } else {
            // ---- candidates: every init before the records' wait (unpack runs it in the stall) ----
            reconfig_data_format_srca(cb_stats);
            pack_reconfig_data_format(cb_rms_root);
            if constexpr (VARIANT == 10) {
                ff_init<2>();
            } else if constexpr (VARIANT == 1 || VARIANT == 5) {
                copy_init(cb_stats);
            } else {
                load_init<(VARIANT >= 6 ? 4 : VARIANT)>();
            }
            if constexpr (VARIANT == 9) {
                UNPACK((config_unpacker_x_end<p_setadc::UNP_A>(2)));  // 32 datums per UNPACR
                PACK((llk_pack_init_impl<PackMode::Default, false, false, false>(
                    pack_src_format[get_output_id(cb_rms_root)], 2, 32, 1, 1, false)));  // rows 0-1 of face 0: 128 B
            }
            if constexpr (VARIANT == 6 || VARIANT == 8) {
                // row-only pack: face_r_dim 1, 2 faces -> row 0 of faces 0 / 1 packed contiguously (the joined 128 B
                // record at the page start; the op's reader joins it by a RISC-V copy today)
                PACK((llk_pack_init_impl<PackMode::Default, false, false, false>(
                    pack_src_format[get_output_id(cb_rms_root)], 1, 32, 2, 1, false)));
            }
            add_binary_tile_init();
            rsqrt_tile_init();
            cb_reserve_back(cb_rms_root, no);
            {
                FF_ZONE("cmp_wait_stats");
                cb_wait_front(cb_stats, G * Q);
            }
            FF_ZONE("cmp_fold");
            for (uint32_t r = 0; r < no; ++r) {
                tile_regs_acquire();
                if constexpr (VARIANT == 1 || VARIANT == 5) {
                    copy_tile(cb_stats, r, 0);
                    for (uint32_t pos = 1; pos < G; ++pos) {
                        copy_tile(cb_stats, pos * Q + r, 1);
                        if constexpr (VARIANT == 1) {
                            add_binary_tile(0, 1, 0);
                        } else {
                            MATH((_llk_math_eltwise_binary_sfpu_params_(
                                ckernel::sfpu::calculate_sfpu_binary<
                                    APPROX,
                                    ckernel::BinaryOp::ADD,
                                    2 /*ITERATIONS: rows 0-3 of each face*/,
                                    DST_ACCUM_MODE>,
                                0,
                                1,
                                0,
                                ckernel::VectorMode::R)));
                        }
                    }
                    MATH((_llk_math_eltwise_unary_sfpu_params_(
                        finalize_row0_body, 0, ckernel::VectorMode::R, inv_w_bits, eps_bits)));
                } else {
                    if constexpr (VARIANT == 10) {
                        ff_fold_chunks<2, G, Q, 0>(r, inv_w_bits, eps_bits);
                    } else if constexpr (VARIANT == 9) {
                        fold_compact<G>(r, inv_w_bits, eps_bits);
                    } else if constexpr (VARIANT >= 7) {
                        fold_multi<G, Q>(r, inv_w_bits, eps_bits);
                    } else {
                        fold_chunks<(VARIANT == 6 ? 4 : VARIANT), G, Q, 0>(r, inv_w_bits, eps_bits);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rms_root);
                tile_regs_release();
            }
            cb_push_back(cb_rms_root, no);
            if constexpr (VARIANT == 6 || VARIANT == 8 || VARIANT == 9 || VARIANT == 10) {
                PACK((llk_pack_init(cb_rms_root)));  // back to the full-tile MOP (after the push: off the chain)
            }
            cb_pop_front(cb_stats, G * Q);
        }
    }
}
