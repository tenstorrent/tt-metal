// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Untilize of a 1-tile-wide TILE block (32-channel slab slices), Blackhole, RAW LLK.
//
// Bypassed helper: compute_kernel_lib::untilize<1, ...> (ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl).
// At block_width_tiles == 1 the helper cannot take fast_untilize (BH needs >= 2 tiles), so it runs one
// pack_untilize_block<1, 1> per tile-row: one CB wait/reserve/pop/push, one DEST section (math/pack semaphore
// handshake) holding 1 of the 4 tiles a fp32 half-sync section holds, and the regular untilize pack MOP, which per
// 32-datum output row issues ADDRCRZW + INCADCZW + PACR(Last=1) + CFGSHIFTMASK(L1_Dest_addr += 64 B) + NOP.
// Measured (1 core, bf8_b -> bf16, fp32 DEST): 157 ns/tile, packer-bound (pack-only 154 ns, unpack+math 34-40 ns).
//
// What this file changes (same unpack, same A2D datacopy, same packer formats; output bit-identical):
//   * `rows_per_section` tile-rows per DEST section (<= 4 at fp32 half-sync): one handshake and one CB
//     wait/reserve/pop/push per section (or one CB sync per caller batch of `batch_rows` tile-rows);
//   * Pack::RowMop   keeps the regular untilize MOP (llk_pack_untilize_impl, block_rt_dim = rows_per_section);
//     Pack::Contig   packs the section as ONE contiguous L1 stream: L1_Dest_addr programmed once per section,
//                    16 identical PACRs (TWO_INTFS, y_src += 1, Last=0) per face pair, replayed from the pack
//                    replay buffer, Last=1 only on the section's final PACR. A 1-wide untilize writes row r of
//                    tile-row t at byte (t * 32 + r) * 64, exactly the order the PACRs emit, so no per-row address
//                    update, W reload or Last is needed (42.6 ns/tile, within 7% of the unpack+math floor);
//     Pack::Strided  Last=1 per row like the regular MOP, L1 row advance from the PAC ch1 Y counter (64 B Y
//                    stride, the BH fast-untilize strided mechanism) instead of a cfg write: 87 ns/tile.
//
// Exit-state contract (the bug this experiment hit): leave the PAC ch0 Y counter at 0, as _llk_pack_untilize_ does
// after every face pair. The BH PackMode::Tilize pack (tilize of 1-tile-wide rows, which conv2d_3x3's tap-major im2col
// runs right after the untilize) does not reset Y at tile start: a Y left at 16 reads the first tilized tile's face 0
// 8 rows low. Output stays PCC 0.9999 but is wrong (max abs err 26 on a 288-term conv): only a bit-exact gate sees it.
//
// The helper cannot express either change: its loop is one LLK call per tile-row, and no LLK exposes a contiguous
// 1-wide pack stream (fast_untilize's contiguous stream exists only for 2-4-tile rows).

#pragma once

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

namespace narrow_untilize {

enum class Pack : uint32_t { RowMop = 0, Contig = 1, Strided = 2 };
// Diagnostics only (wrong output by construction): NoPack skips the PACRs, PackOnly skips unpack + datacopy.
enum class Diag : uint32_t { None = 0, NoPack = 1, PackOnly = 2 };

#ifdef TRISC_PACK
constexpr uint32_t kReplayOffset = ckernel::packer::replay_buf_offset;
constexpr uint32_t kRowsPerFacePair = 16;

// One output row of a 1-wide tile: interfaces 0/1 read face-row y of F0|F1 (or F2|F3), 32 datums; ADDR_MOD_1
// (programmed by the untilize init: y_src += 1) moves to the next face row.
#define NU_PACR_ROW(last)                \
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

// Pack::Strided row: Last=1, ADDR_MOD_2 advances y_src and the ch1 (L1-side) Y counter.
#define NU_PACR_ROW_STRIDED              \
    TTI_PACR(                            \
        p_pacr::CFG_CTXT_0,              \
        p_pacr::NO_ROW_PAD_ZERO,         \
        p_pacr::DST_ACCESS_STRIDED_MODE, \
        ADDR_MOD_2,                      \
        p_pacr::ADDR_CNT_CTXT_0,         \
        0,                               \
        p_pacr::TWO_INTFS_ACTIVE,        \
        0,                               \
        0,                               \
        p_pacr::NO_CTXT_CTRL,            \
        0,                               \
        1)

inline void contig_init() {
    load_replay_buf(kReplayOffset, kRowsPerFacePair, [] {
        for (uint32_t i = 0; i < kRowsPerFacePair; ++i) {
            NU_PACR_ROW(0);
        }
    });
}

// Packs DEST tiles [0, g) of the current section to L1 at `addr` (16 B units, packer convention) as one stream.
inline void pack_contig(uint32_t addr, uint32_t g) {
    program_packer_destination(addr);
    for (uint32_t t = 0; t < g; ++t) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);               // ch0 z = 0: F0|F1
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t);  // DEST tile t of the section
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        lltt::replay(kReplayOffset, kRowsPerFacePair);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);  // z += 2 faces: F2|F3
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        if (t + 1 < g) {
            lltt::replay(kReplayOffset, kRowsPerFacePair);
        } else {
            lltt::replay(kReplayOffset, kRowsPerFacePair - 1);
            NU_PACR_ROW(1);  // closes the section's stream
        }
    }
    // Exit state of _llk_pack_untilize_: ch0 Y = 0 (see the header comment), Z cleared, W = 0.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
}

inline void strided_init(uint32_t out_row_bytes) {
    addr_mod_pack_t{.y_src = {.incr = 1}, .y_dst = {.incr = 1}}.set(ADDR_MOD_2);
    TT_SETDMAREG(0, LOWER_HALFWORD(out_row_bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(out_row_bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_NOP;
    load_replay_buf(kReplayOffset, kRowsPerFacePair, [] {
        for (uint32_t i = 0; i < kRowsPerFacePair; ++i) {
            NU_PACR_ROW_STRIDED;
        }
    });
}

inline void strided_uninit() {
    // ch1 strides back to their zero baseline (set_packer_strides owns it), ch1 Y counter to 0.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_NOP;
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1000);
}

inline void pack_strided(uint32_t addr, uint32_t g) {
    program_packer_destination(addr);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1000);  // ch1 y = 0: output row 0 of the section
    for (uint32_t t = 0; t < g; ++t) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        lltt::replay(kReplayOffset, kRowsPerFacePair);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        lltt::replay(kReplayOffset, kRowsPerFacePair);
    }
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1000);
}
#endif

// Init for untilize_1wide: the helper's default reconfig + pack_untilize_init<1, 1>, plus the pack-mode setup.
template <uint32_t icb, uint32_t ocb, Pack pm>
ALWI void init() {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    pack_untilize_init<1, 1>(icb, ocb);
    if constexpr (pm == Pack::Contig) {
        PACK((contig_init()));
    } else if constexpr (pm == Pack::Strided) {
        PACK((strided_init(64)));
    }
}

template <uint32_t ocb, Pack pm>
ALWI void uninit() {
    if constexpr (pm == Pack::Strided) {
        PACK((strided_uninit()));
    }
    pack_untilize_uninit(ocb);
}

// One DEST section: tile-rows [tile0, tile0 + g) of the fronted input to output pages [tile0, tile0 + g).
template <uint32_t icb, uint32_t ocb, Pack pm, Diag diag>
ALWI void section(uint32_t tile0, uint32_t g) {
    MATH((llk_math_wait_for_dest_available()));
    if constexpr (diag != Diag::PackOnly) {
        for (uint32_t i = 0; i < g; ++i) {
            UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
                icb, tile0 + i)));
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    i, icb)));
        }
    }
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    PACK((llk_packer_wait_for_math_done()));
#ifdef TRISC_PACK
    if constexpr (diag != Diag::NoPack) {
        const uint32_t out_id = get_output_id(ocb);
        const auto& cbi = get_local_cb_interface(out_id);
        const uint32_t addr = cbi.fifo_wr_ptr - 1 + tile0 * cbi.fifo_page_size;
        if constexpr (pm == Pack::Contig) {
            pack_contig(addr, g);
        } else if constexpr (pm == Pack::Strided) {
            pack_strided(addr, g);
        } else {
            llk_pack_untilize_impl<1, 1>(
                g,
                addr,
                pack_src_format[out_id],
                pack_dst_format[out_id],
                cbi.fifo_page_size,
                get_output_face_r_dim(out_id),
                get_output_num_faces(out_id),
                0,
                0);
        }
    }
#endif
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}

// Untilizes num_rows 1-tile tile-rows from icb into ocb (one 2 KB bf16 page per tile-row), one DEST section per
// rows_per_section tile-rows. Sections never straddle a batch_rows boundary, so a CB whose capacity is a whole number
// of batches never wraps inside a section. CB sync per section (default: the consumer sees rows as soon as they land,
// like the helper's per-row push) or, with per_batch_sync, once per batch. Includes init and uninit.
template <
    uint32_t icb,
    uint32_t ocb,
    Pack pm,
    uint32_t rows_per_section,
    Diag diag = Diag::None,
    bool per_batch_sync = false,
    typename InDFB,
    typename OutDFB>
ALWI void untilize_1wide(InDFB& in, OutDFB& out, uint32_t num_rows, uint32_t batch_rows) {
    static_assert(rows_per_section >= 1 && rows_per_section <= (DST_ACCUM_MODE ? 4 : 8), "one DEST half");
    init<icb, ocb, pm>();
    for (uint32_t done = 0; done < num_rows; done += batch_rows) {
        const uint32_t n = num_rows - done < batch_rows ? num_rows - done : batch_rows;
        if constexpr (per_batch_sync) {
            in.wait_front(n);
            out.reserve_back(n);
        }
        for (uint32_t r = 0; r < n; r += rows_per_section) {
            const uint32_t g = n - r < rows_per_section ? n - r : rows_per_section;
            if constexpr (per_batch_sync) {
                section<icb, ocb, pm, diag>(r, g);
            } else {
                in.wait_front(g);
                out.reserve_back(g);
                section<icb, ocb, pm, diag>(0, g);
                in.pop_front(g);
                out.push_back(g);
            }
        }
        if constexpr (per_batch_sync) {
            in.pop_front(n);
            out.push_back(n);
        }
    }
    uninit<ocb, pm>();
}

// Wide slices (2+ tiles, already on BH fast_untilize): the helper's fast path, CB sync per batch instead of per row.
template <uint32_t ct, uint32_t icb, uint32_t ocb, typename InDFB, typename OutDFB>
ALWI void untilize_wide_batched(InDFB& in, OutDFB& out, uint32_t num_rows, uint32_t batch_rows) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    fast_untilize_init<ct>(icb, ocb);
    for (uint32_t done = 0; done < num_rows; done += batch_rows) {
        const uint32_t n = num_rows - done < batch_rows ? num_rows - done : batch_rows;
        in.wait_front(n * ct);
        out.reserve_back(n * ct);
        for (uint32_t r = 0; r < n; ++r) {
            fast_untilize_block<ct>(icb, ocb, r * ct, r * ct);
        }
        in.pop_front(n * ct);
        out.push_back(n * ct);
    }
    fast_untilize_uninit<ct>(ocb);
}

}  // namespace narrow_untilize
