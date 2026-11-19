// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (compute only, one core, Blackhole, fp32 DEST half-sync, LoFi): conv2d_3x3's N-slice prelude for the
// shifted taps (kw = 0 / 2) of a 1-tile-wide channel slice (res4.x: W = 64, C_slice 32, bf8_b).
//
// Geometry: an image of R rows x W pixels (W % 32 == 0, tpr = W / 32 tiles per row), its whole TILE slab resident in
// CB_SRC (tile k = image row k / tpr, column k % tpr). Output tile-rows ti in [t0, t0 + bm): for each kernel row kh
// (input row h = ti / tpr + kh * d - d) the taps kw = 0 and kw = 2 (pixel shift -d / +d), in the op's unit order
// (tile-row, kernel row, tap). A tap whose row is outside the image is the zero tile.
//
// variant 0  (baseline, the op today): untilize the slab into the padded row-major slab (row slot pitch W + d, d zero
//            pixels in front of every row: conv2d_3x3_narrow_untilize.hpp untilize_1wide_at's section and stream), then
//            padded_tilize's 1-wide tile loop (NarrowTilizeBatch: 4 tilizes per DEST section) from computed sources.
// variant 1  baseline tilize only (the untilize runs once, before the reps).
// variant 2  baseline untilize only.
// variant 10 (candidate): every shifted tap straight from the TILE slab. Unpack AB puts the tap's own tile ("cur",
//            the same image row) in SrcA and its neighbour across the shift ("nb": the next tile of the row for kw = 2,
//            the previous for kw = 0, or the zero tile at the row's end) in SrcB; the math thread builds the shifted
//            tile in DEST with 64 one-row moves (MOVA2D from SrcA, MOVB2D from SrcB), one per DEST row, then the
//            standard pack writes it: 4 taps per DEST section, one CB reserve / push per section. The values move
//            unchanged (bf8_b -> 19-bit Src -> fp32 DEST -> bf8_b, each 16-datum row keeps its datums, so its shared
//            exponent too): bit-identical to the untilize / tilize round trip.
// variant 11 diag (wrong output): unpack A(cur) + standard A2D datacopy + pack, 4 per section: the per-tile floor of
//            any DEST-side shift.
// variant 12 diag (wrong output): as 10 without the moves (unpack AB, release the banks, pack the cleared DEST).
//
// RAW LLK (bench): the shifted-tap path has no helper (compute_kernel_lib::tilize / untilize read at CB pointers and
// have no row shift); the baseline is the op's own raw loops.

#include <cstdint>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t CB_SRC = 0;
constexpr uint32_t CB_ZERO = 1;
constexpr uint32_t CB_RM = 2;
constexpr uint32_t CB_OUT = 16;

// ---- baseline: the op's 1-wide untilize stream (conv2d_3x3_narrow_untilize.hpp, copied) ----
#ifdef TRISC_PACK
constexpr uint32_t kReplayOffset = ckernel::packer::replay_buf_offset;
constexpr uint32_t kRowsPerFacePair = 16;
#define BENCH_NU_PACR_ROW(last)          \
    TTI_PACR(                            \
        p_pacr::CFG_CTXT_0,              \
        p_pacr::NO_ROW_PAD_ZERO,         \
        p_pacr::DST_ACCESS_STRIDED_MODE, \
        ADDR_MOD_1,                      \
        p_pacr::ADDR_CNT_CTXT_0,         \
        0,                               \
        p_pacr::TWO_INTFS_ACTIVE,        \
        0,                               \
        0,                               \
        p_pacr::NO_CTXT_CTRL,            \
        0,                               \
        last)
inline void contig_init() {
    load_replay_buf(kReplayOffset, kRowsPerFacePair, [] {
        for (uint32_t i = 0; i < kRowsPerFacePair; ++i) {
            BENCH_NU_PACR_ROW(0);
        }
    });
}
inline void pack_contig_range(uint32_t addr, uint32_t t0, uint32_t t1) {
    program_packer_destination(addr);
    for (uint32_t t = t0; t < t1; ++t) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        lltt::replay(kReplayOffset, kRowsPerFacePair);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        if (t + 1 < t1) {
            lltt::replay(kReplayOffset, kRowsPerFacePair);
        } else {
            lltt::replay(kReplayOffset, kRowsPerFacePair - 1);
            BENCH_NU_PACR_ROW(1);
        }
    }
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
}
#endif

#ifdef TRISC_UNPACK
inline void direct_unpack_tilize(uint32_t src) {
    const uint32_t id = get_operand_id(CB_RM);
    llk_unpack_tilize_impl(
        (src >> 4) - 1,
        0,
        unpack_src_format[id],
        unpack_dst_format[id],
        get_operand_face_r_dim(id),
        get_operand_num_faces(id),
        get_operand_narrow_tile(id));
}
#endif

// ---- candidate: one-row moves ----
// DEST row i of a 32x32 tile (fp32, 64 rows): faces F0 [0, 16) pixels 0..15 channels 0..15, F1 [16, 32) pixels 0..15
// channels 16..31, F2 [32, 48) pixels 16..31 channels 0..15, F3 [48, 64). SrcA / SrcB hold an unpacked tile the same
// way.
constexpr int pix_of_row(uint32_t i) { return static_cast<int>(i < 32 ? (i % 16) : 16 + (i % 16)); }
// Row offset of pixel p inside its channel half (the half's own +16 cancels in a row difference).
constexpr int fpos(int p) { return p < 16 ? p : p + 16; }
// Output pixel r takes input pixel r + s of cur, or of nb when that leaves the tile (|s| < 32).
constexpr bool from_nb(int s, uint32_t i) {
    const int p = pix_of_row(i) + s;
    return p < 0 || p >= 32;
}
// The move's source row immediate with the A, B and D counters all at i (each move adds 1 to all three).
constexpr uint32_t imm_of(int s, uint32_t i) {
    const int r = pix_of_row(i);
    int p = r + s;
    p = p < 0 ? p + 32 : (p >= 32 ? p - 32 : p);
    return static_cast<uint32_t>((fpos(p) - fpos(r)) & 63);
}

#ifdef TRISC_UNPACK
// After llk_unpack_AB_init: each operand's whole tile in ONE unpack (X end = 1024 datums: faces F0..F3 land in Src rows
// 0..63 of one bank, one bank flip per tile), as the BH matmul's full-tile unpack. The stock AB MOP unpacks face by
// face with a flip per face (the eltwise math clears per face), so one Src bank never holds the whole tile.
inline void full_tile_ab_init() {
    TT_SETADCXX(p_setadc::UNP_A, 32 * 32 - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, 32 * 32 - 1, 0x0);
    ckernel::ckernel_template tmp(
        1,
        1,
        TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
#endif

#ifdef TRISC_UNPACK
// SrcB only: one full-tile UNPACR on unpacker 1 per _llk_unpack_AB_ call (its SrcA address is ignored).
inline void full_tile_b_init() {
    TT_SETADCXX(p_setadc::UNP_B, 32 * 32 - 1, 0x0);
    ckernel::ckernel_template tmp(1, 1, TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
#endif

// SrcB row of DEST row i's source (an unpacked tile's own layout), for shift s.
constexpr uint32_t srcrow(int s, uint32_t i) { return (imm_of(s, i) + i) & 63; }

#ifdef TRISC_MATH
// SrcB path: every row a one-row MOVB2D with explicit SrcB / DEST row immediates (counters stay 0). One-row MOVA2D is
// not usable here: with fp32 DEST it leaves the rows of faces F1 / F3 unwritten (measured), while 8-row MOVA2D and
// one-row MOVB2D are exact.
constexpr uint32_t ADDR_MOD_ZERO = ADDR_MOD_7;
template <int s, bool nb, uint32_t i>
ALWI void movb_row() {
    if constexpr (from_nb(s, i) == nb) {
        TTI_MOVB2D(0, srcrow(s, i), ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, i);
    }
}
template <int s, bool nb, uint32_t... I>
ALWI void movb_pass(std::integer_sequence<uint32_t, I...>) {
    (movb_row<s, nb, I>(), ...);
}
inline void srcb_math_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_ZERO);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// Replayed SrcB path (d < 16): runs of consecutive rows with the same source offset are one replay of a recorded run of
// one-row MOVB2D that step the SrcB and DEST counters by 1 (ADDR_MOD_STEP). Rows 0..15 and 16..31 (F0, F1) have the
// same offsets, as do 32..47 and 48..63 (F2, F3). kw = 2 (s = +d): F0/F1 rows r < 16 - d from cur row r + d (offset d),
// the next d rows from cur F2/F3 (offset 16 + d); F2/F3 rows r < 16 - d offset d, the last d rows from nb. kw = 0
// (s = -d): F0/F1 rows r < d from nb, the rest offset 64 - d (mod 64); F2/F3 rows r < d from cur F0/F1 (offset 48 - d),
// the rest offset 64 - d. Replay buffer [0, 32): R1 = offset d x (16 - d), R2 = 16 + d x d, R3 = 64 - d x (16 - d),
// R4 = 48 - d x d. nb rows (2d per tile) are explicit one-row moves.
constexpr uint32_t ADDR_MOD_STEP = ADDR_MOD_6;
template <int d>
inline void replay_math_init() {
    srcb_math_init();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 1}, .dest = {.incr = 1}}.set(ADDR_MOD_STEP);
    // d >= 16: not replayed (the explicit SrcB path, variant 10, covers it); compiled empty.
    if constexpr (d > 0 && d < 16) {
        load_replay_buf(0, 32, [] {
            for (int k = 0; k < 16 - d; ++k) {
                TTI_MOVB2D(0, d, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < d; ++k) {
                TTI_MOVB2D(0, 16 + d, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < 16 - d; ++k) {
                TTI_MOVB2D(0, 64 - d, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < d; ++k) {
                TTI_MOVB2D(0, 48 - d, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
        });
    }
}
// cur rows of DEST tile (set by set_dst_write_addr), counters from 0; leaves them dirty (caller resets).
template <int d, bool kw2>
ALWI void replay_cur_pass() {
    constexpr uint32_t R1 = 0, R2 = 16 - d, R3 = 16, R4 = 32 - d;
    if constexpr (d >= 16) {
    } else if constexpr (kw2) {
        lltt::replay(R1, 16 - d);
        lltt::replay(R2, d);
        lltt::replay(R1, 16 - d);
        lltt::replay(R2, d);
        lltt::replay(R1, 16 - d);
        TTI_INCRWC(0, d, d, 0);
        lltt::replay(R1, 16 - d);
    } else {
        TTI_INCRWC(0, d, d, 0);
        lltt::replay(R3, 16 - d);
        TTI_INCRWC(0, d, d, 0);
        lltt::replay(R3, 16 - d);
        lltt::replay(R4, d);
        lltt::replay(R3, 16 - d);
        lltt::replay(R4, d);
        lltt::replay(R3, 16 - d);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
// 1:1 diag: 64 rows, 4 replays of the 16-long run recorded at offset 0 for s = 0.
inline void replay_copy_init() {
    srcb_math_init();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 1}, .dest = {.incr = 1}}.set(ADDR_MOD_STEP);
    load_replay_buf(0, 16, [] {
        for (int k = 0; k < 16; ++k) {
            TTI_MOVB2D(0, 0, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
        }
    });
}
// One pass of DEST tile j's rows from the SrcB tile (cur rows, or nb rows), then SrcB back to the unpacker.
template <int s, bool nb>
ALWI void srcb_pass() {
    movb_pass<s, nb>(std::make_integer_sequence<uint32_t, 64>{});
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
#endif

#ifdef TRISC_MATH
constexpr uint32_t ADDR_MOD_SHIFT = ADDR_MOD_6;
// force: 0 = by the shift, 1 = every row MOVA2D (SrcA), 2 = every row MOVB2D (SrcB) (diags, s = 0).
template <int s, uint32_t i, int force = 0>
ALWI void move_row() {
    if constexpr (force == 2 || (force == 0 && from_nb(s, i))) {
        TTI_MOVB2D(0, imm_of(s, i), ADDR_MOD_SHIFT, p_movb2d::MOV_1_ROW, 0);
    } else {
        TTI_MOVA2D(0, imm_of(s, i), ADDR_MOD_SHIFT, p_mova2d::MOV_1_ROW, 0);
    }
}
template <int s, int force, uint32_t... I>
ALWI void move_tile(std::integer_sequence<uint32_t, I...>) {
    (move_row<s, I, force>(), ...);
}
inline void shift_math_init() {
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 1}, .dest = {.incr = 1}}.set(ADDR_MOD_SHIFT);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
// DEST tile j <- the shifted tap (SrcA = cur, SrcB = nb, both valid from the unpacker).
template <int s, int force = 0>
ALWI void shift_tile(uint32_t j) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(j);
    if constexpr (force == 3) {
        // 8-row aligned 1:1 copy of SrcA
        for (uint32_t r = 0; r < 64; r += 8) {
            TT_MOVA2D(0, r, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, r);
        }
    } else {
        move_tile<s, force>(std::make_integer_sequence<uint32_t, 64>{});
    }
    // Both banks back to the unpacker, counters to 0.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t reps = get_compile_time_arg_val(1);
    constexpr int32_t d = get_compile_time_arg_val(2);
    constexpr int32_t R = get_compile_time_arg_val(3);
    constexpr uint32_t t0 = get_compile_time_arg_val(4);
    constexpr uint32_t bm = get_compile_time_arg_val(5);
    constexpr int32_t W = get_compile_time_arg_val(6);
    constexpr int32_t tpr = W / 32;
    constexpr uint32_t S = R * tpr;  // slab tiles
    constexpr int32_t slot_px = W + d;
    constexpr uint32_t n_out = bm * 6;
    static_assert(d > 0 && d < 32 && W % 32 == 0);

    compute_kernel_hw_startup(CB_SRC, CB_OUT);
    DataflowBuffer out(CB_OUT);

    [[maybe_unused]] uint32_t src_l1 = 0, zero_l1 = 0, rm_l1 = 0, rm_zero_l1 = 0;
    UNPACK((src_l1 = get_local_cb_interface(get_operand_id(CB_SRC)).fifo_rd_ptr << 4));
    UNPACK((zero_l1 = get_local_cb_interface(get_operand_id(CB_ZERO)).fifo_rd_ptr << 4));
    UNPACK((rm_l1 = get_local_cb_interface(get_operand_id(CB_RM)).fifo_rd_ptr << 4));
    PACK((rm_l1 = get_local_cb_interface(get_output_id(CB_RM)).fifo_wr_ptr << 4));
    // 32 zero pixels after the last slot's trailing gap (never written).
    rm_zero_l1 = rm_l1 + static_cast<uint32_t>(R * slot_px + d) * 64;
    rm_zero_l1 = (rm_zero_l1 + 63) & ~63u;

    auto untilize = [&]() {
        MaybeDeviceZoneScope("bench_untilize");
        reconfig_data_format_srca(CB_SRC);
        pack_reconfig_data_format(CB_RM);
        pack_untilize_init<1, 1>(CB_SRC, CB_RM);
        PACK((contig_init()));
        for (uint32_t k0 = 0; k0 < S; k0 += 4) {
            const uint32_t g = S - k0 < 4 ? S - k0 : 4;
            MATH((llk_math_wait_for_dest_available()));
            for (uint32_t i = 0; i < g; ++i) {
                UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
                    CB_SRC, k0 + i)));
                MATH((llk_math_eltwise_unary_datacopy<
                      DataCopyType::A2D,
                      DST_ACCUM_MODE,
                      BroadcastType::NONE,
                      UnpackToDestEn>(i, CB_SRC)));
            }
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_packer_wait_for_math_done()));
#ifdef TRISC_PACK
            // tile k -> slot of row k / tpr, column k % tpr: runs of a row's tiles are contiguous.
            for (uint32_t t = 0; t < g;) {
                const uint32_t k = k0 + t;
                const uint32_t row = k / tpr;
                uint32_t e = t + 1;
                while (e < g && (k0 + e) / tpr == row) {
                    ++e;
                }
                const uint32_t px = row * slot_px + d + (k % tpr) * 32;
                pack_contig_range(((rm_l1 + px * 64) >> 4) - 1, t, e);
                t = e;
            }
#endif
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        }
        pack_untilize_uninit(CB_RM);
    };

    auto tilize = [&]() {
        MaybeDeviceZoneScope("bench_tilize");
        reconfig_data_format_srca(CB_RM);
        pack_reconfig_data_format(CB_OUT);
        fast_tilize_init(CB_RM, 1, CB_OUT);
        uint32_t j = 0;
        for (uint32_t t = 0; t < bm; ++t) {
            const int32_t ti = t0 + t;
            const int32_t ho = ti / tpr;
            const int32_t wo0 = (ti % tpr) * 32;
            for (int32_t kh = 0; kh < 3; ++kh) {
                const int32_t h = ho + kh * d - d;
                for (int32_t kw = 0; kw < 3; kw += 2) {
                    [[maybe_unused]] uint32_t src =
                        (h < 0 || h >= R) ? rm_zero_l1 : rm_l1 + static_cast<uint32_t>(h * slot_px + wo0 + kw * d) * 64;
                    if (j == 0) {
                        MATH((llk_math_wait_for_dest_available()));
                        PACK((llk_packer_wait_for_math_done()));
                    }
                    UNPACK((direct_unpack_tilize(src)));
                    MATH((llk_math_eltwise_unary_datacopy<
                          DataCopyType::A2D,
                          DST_ACCUM_MODE,
                          BroadcastType::NONE,
                          UnpackToDestEn>(j, CB_RM)));
                    if (++j == 4) {
                        out.reserve_back(4);
                        for (uint32_t i = 0; i < 4; ++i) {
                            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, CB_OUT, i)));
                        }
                        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                        out.push_back(4);
                        j = 0;
                    }
                }
            }
        }
        if (j > 0) {
            out.reserve_back(j);
            for (uint32_t i = 0; i < j; ++i) {
                PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, CB_OUT, i)));
            }
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            out.push_back(j);
        }
        fast_tilize_uninit(CB_RM, CB_OUT, 1);
    };

    // Candidate (and its diags): taps straight from the TILE slab.
    auto shift = [&]() {
        MaybeDeviceZoneScope("bench_shift");
        reconfig_data_format(CB_SRC, CB_SRC);
        pack_reconfig_data_format(CB_OUT);
        if constexpr (variant == 11) {
            copy_init(CB_SRC);
        } else if constexpr (variant == 27) {
        } else if constexpr (
            variant == 10 || variant == 13 || variant == 23 || variant == 24 || variant == 25 || variant == 28) {
            UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(CB_SRC, CB_SRC)));
            UNPACK((full_tile_b_init()));
            if constexpr (variant == 24) {
                MATH((replay_copy_init()));
            } else if constexpr (variant == 25) {
                MATH((replay_math_init<d>()));
            } else {
                MATH((srcb_math_init()));
            }
        } else {
            UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(CB_SRC, CB_SRC)));
            UNPACK((full_tile_ab_init()));
            MATH((shift_math_init()));
        }
        pack_init(CB_OUT);
        uint32_t j = 0;
        for (uint32_t t = 0; t < bm; ++t) {
            const int32_t ti = t0 + t;
            const int32_t ho = ti / tpr;
            const int32_t c = ti % tpr;
            for (int32_t kh = 0; kh < 3; ++kh) {
                const int32_t h = ho + kh * d - d;
                const bool in_img = h >= 0 && h < R;
                for (int32_t kw = 0; kw < 3; kw += 2) {
                    if (j == 0) {
                        MATH((llk_math_wait_for_dest_available()));
                        PACK((llk_packer_wait_for_math_done()));
                    }
                    [[maybe_unused]] uint32_t a = zero_l1, b = zero_l1;
                    if (in_img) {
                        const int32_t cn = kw == 0 ? c - 1 : c + 1;
                        a = src_l1 + static_cast<uint32_t>(h * tpr + c) * 1088;
                        b = (cn < 0 || cn >= tpr) ? zero_l1 : src_l1 + static_cast<uint32_t>(h * tpr + cn) * 1088;
                    }
                    if constexpr (variant == 27) {
                    } else if constexpr (variant == 28) {
                        UNPACK((_llk_unpack_AB_<BroadcastType::NONE>((a >> 4) - 1, (a >> 4) - 1)));
                        // SETRWC does not wait for SrcB valid: one move first (it waits), then release.
                        MATH((TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, 0)));
                        MATH((TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD)));
                    } else if constexpr (
                        variant == 10 || variant == 13 || variant == 23 || variant == 24 || variant == 25) {
                        const int32_t cn = kw == 0 ? c - 1 : c + 1;
                        const bool nb_zero = !in_img || cn < 0 || cn >= tpr;
                        const bool skip_cur = variant == 13 && !in_img;
                        const bool skip_nb = variant == 13 && nb_zero;
                        MATH((math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(j)));
                        if (!skip_cur) {
                            UNPACK((_llk_unpack_AB_<BroadcastType::NONE>((a >> 4) - 1, (a >> 4) - 1)));
#ifdef TRISC_MATH
                            if constexpr (variant == 23) {
                                srcb_pass<0, false>();
                            } else if constexpr (variant == 24) {
                                lltt::replay(0, 16);
                                lltt::replay(0, 16);
                                lltt::replay(0, 16);
                                lltt::replay(0, 16);
                                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                            } else if constexpr (variant == 25) {
                                if (kw == 0) {
                                    replay_cur_pass<d, false>();
                                } else {
                                    replay_cur_pass<d, true>();
                                }
                                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                            } else if (kw == 0) {
                                srcb_pass<-d, false>();
                            } else {
                                srcb_pass<d, false>();
                            }
#endif
                        }
                        if (!skip_nb && variant != 23 && variant != 24) {
                            UNPACK((_llk_unpack_AB_<BroadcastType::NONE>((b >> 4) - 1, (b >> 4) - 1)));
#ifdef TRISC_MATH
                            if (kw == 0) {
                                srcb_pass<-d, true>();
                            } else {
                                srcb_pass<d, true>();
                            }
#endif
                        }
                    } else if constexpr (variant == 11) {
                        if (in_img) {
                            copy_tile(CB_SRC, static_cast<uint32_t>(h * tpr + c), j);
                        } else {
                            copy_tile(CB_ZERO, 0, j);
                        }
                    } else {
                        UNPACK((_llk_unpack_AB_<BroadcastType::NONE>((a >> 4) - 1, (b >> 4) - 1)));
#ifdef TRISC_MATH
                        if constexpr (variant == 12) {
                            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
                        } else if constexpr (variant == 20) {
                            shift_tile<0, 1>(j);
                        } else if constexpr (variant == 21) {
                            shift_tile<0, 2>(j);
                        } else if constexpr (variant == 22) {
                            shift_tile<0, 3>(j);
                        } else if (kw == 0) {
                            shift_tile<-d>(j);  // variant 14: SrcA cur + SrcB nb (one-row MOVA2D: broken, see above)
                        } else {
                            shift_tile<d>(j);
                        }
#endif
                    }
                    if (++j == 4) {
                        out.reserve_back(4);
                        for (uint32_t i = 0; i < 4; ++i) {
                            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, CB_OUT, i)));
                        }
                        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                        out.push_back(4);
                        j = 0;
                    }
                }
            }
        }
        if (j > 0) {
            out.reserve_back(j);
            for (uint32_t i = 0; i < j; ++i) {
                PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, CB_OUT, i)));
            }
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            out.push_back(j);
        }
        MATH((math::clear_dst_reg_addr()));
    };

    if constexpr (variant == 1) {
        untilize();
    }
    for (uint32_t rep = 0; rep < reps; ++rep) {
        if constexpr (variant == 0) {
            untilize();
            tilize();
        } else if constexpr (variant == 1) {
            tilize();
        } else if constexpr (variant == 2) {
            untilize();
        } else {
            shift();
        }
        if constexpr (variant != 2) {
            cb_wait_front(CB_OUT, n_out);
            cb_pop_front(CB_OUT, n_out);
        }
    }
}
