// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Untilize of a 1-tile-wide TILE block (32-channel slab slices), Blackhole. RAW LLK: bypasses
// compute_kernel_lib::untilize<1, ...> (untilize_helpers.inl), on a measured win.
//
// At block_width_tiles == 1 the helper cannot take fast_untilize (BH needs >= 2 tiles), so it runs one
// pack_untilize_block<1, 1> per tile-row: one CB wait/reserve/pop/push and one DEST section (holding 1 of the 4 tiles a
// fp32 half-sync section holds) per tile-row, and the untilize pack MOP, which closes every 32-datum output row and
// reprograms its L1 address. That path is packer-bound: 157 ns/tile (1 core, bf8_b -> bf16, fp32 DEST), against
// 34-40 ns/tile for unpack + datacopy.
//
// Here, with the same unpack, A2D datacopy and packer formats (output bit-identical):
//   * up to 4 tile-rows per DEST section, with one handshake and one CB wait/reserve/pop/push per section;
//   * the section is packed as ONE contiguous L1 stream. A 1-wide untilize writes row r of tile-row t at byte
//     (t * 32 + r) * 64, exactly the order in which the PACRs emit rows, so the L1 address is programmed once per
//     section, 16 identical PACRs per face pair are replayed from the pack replay buffer, and only the section's
//     last PACR closes the stream.
// Measured: 43.5 ns/tile isolated; res4.0.conv2 @ LoFi untilize zone 9.5 -> 3.0 us, op 44.4 -> 42.3 us.
//
// Exit-state contract: leave the PAC ch0 Y counter at 0, as _llk_pack_untilize_ does after every face pair. The BH
// PackMode::Tilize pack (the 1-tile-wide tilize the tap-major im2col runs next) does not reset Y at tile start, so a Y
// left at 16 reads the first tilized tile's face 0 eight rows low: PCC stays 0.9999, only a bit-exact check sees it.

#pragma once

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

namespace conv2d_3x3::narrow_untilize {

// Tile-rows per DEST section: a fp32 half-sync DEST section holds 4 tiles.
constexpr uint32_t ROWS_PER_SECTION = 4;

#ifdef TRISC_PACK
constexpr uint32_t kReplayOffset = ckernel::packer::replay_buf_offset;
constexpr uint32_t kRowsPerFacePair = 16;

// One output row of a 1-wide tile: interfaces 0/1 read face-row y of F0|F1 (or F2|F3), 32 datums; ADDR_MOD_1
// (programmed by the untilize init: y_src += 1) moves to the next face row.
#define CONV2D_3X3_NU_PACR_ROW(last)     \
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
            CONV2D_3X3_NU_PACR_ROW(0);
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
            CONV2D_3X3_NU_PACR_ROW(1);  // closes the section's stream
        }
    }
    // Exit state of _llk_pack_untilize_ (see the file comment): ch0 Y = 0, Z cleared, W = 0.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
}
#endif

// One DEST section: the g fronted input tiles to the g reserved output pages.
template <uint32_t icb, uint32_t ocb>
ALWI void section(uint32_t g) {
    MATH((llk_math_wait_for_dest_available()));
    for (uint32_t i = 0; i < g; ++i) {
        UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(icb, i)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            i, icb)));
    }
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    PACK((llk_packer_wait_for_math_done()));
#ifdef TRISC_PACK
    const uint32_t out_id = get_output_id(ocb);
    pack_contig(get_local_cb_interface(out_id).fifo_wr_ptr - 1, g);
#endif
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}

// Untilizes num_rows 1-tile tile-rows from icb into ocb (one bf16 page per tile-row), ROWS_PER_SECTION per DEST
// section, with the CB synced per section so the consumer sees rows as they land. Sections never straddle a batch_rows
// boundary, so CBs whose capacity is a whole number of batches never wrap inside a section. Includes init and uninit.
template <uint32_t icb, uint32_t ocb, typename InDFB, typename OutDFB>
ALWI void untilize_1wide(InDFB& in, OutDFB& out, uint32_t num_rows, uint32_t batch_rows) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    pack_untilize_init<1, 1>(icb, ocb);
    PACK((contig_init()));
    for (uint32_t done = 0; done < num_rows; done += batch_rows) {
        const uint32_t n = num_rows - done < batch_rows ? num_rows - done : batch_rows;
        for (uint32_t r = 0; r < n; r += ROWS_PER_SECTION) {
            const uint32_t g = n - r < ROWS_PER_SECTION ? n - r : ROWS_PER_SECTION;
            in.wait_front(g);
            out.reserve_back(g);
            section<icb, ocb>(g);
            in.pop_front(g);
            out.push_back(g);
        }
    }
    pack_untilize_uninit(ocb);
}

}  // namespace conv2d_3x3::narrow_untilize
