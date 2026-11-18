// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated fp32 untilize-pack bench for conv2d_3x3's kw-fold accumulator (compute only, one core).
//
// Each DEST section (fp32 half-sync: 4 tiles) is packed as fp32 rows into a row-major plane, the way kw-fold's
// subblock packs land in ACC. `passes` passes of `sections` sections; with l1acc pass 0 overwrites and later passes
// accumulate (packer L1 accumulation), as kw-fold's Q_1 / Q_0 / Q_2 rounds do. copy = 1 fills DEST with the input's 4
// fp32 tiles (unpack A + datacopy, the correctness run); copy = 0 packs whatever DEST holds (the pack-only timing run).
//
// variant (sbw x sbh subblock = plane width x tile-rows):
//   0 mop2x2   : kw_fold.hpp stream MOP <2>, 2 tile-rows x 2 wide, one stream (stem.conv2 / res2 today)
//   1 mop1x4   : stream MOP <4>, 1 tile-row x 4 wide (N 4 cases today)
//   2 mop4x1   : stream MOP <1>, 4 tile-rows x 1 wide, one stream (center_head.1 today)
//   3 rep4x1   : 4x1, replayed 16-PACR face pairs with explicit ADC moves per face pair (narrow_untilize contig)
//   4 rep4x1_s2: as 3, a stream per 2 tile-rows (an image-row gap mid-section)
//   5 rep4x1_s4: as 3, a stream per tile-row
//   6 tile     : plain tilized fp32 pack of 4 tiles (reference: 4-interface pack, no untilize)
//   7 zst4x1   : 4x1, one replay of 16 PACRs per face pair, the face-pair / tile step folded into ADDR_MOD_2 (y = 0,
//                z += 1: Z stride = 2 faces, so z = 2 is the next tile); no ADC instruction inside the stream
//   8 zst4x1_s2: as 7, a stream per 2 tile-rows
//   9 zst4x1_s4: as 7, a stream per tile-row
//  10 empty    : DEST handshake only (no PACR)
// RAW LLK: bypasses compute_kernel_lib::untilize / pack_untilize_block on purpose (the bench measures the packer's
// stream forms; the helper packs at a CB write pointer, one closed stream per row).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace bench {

#ifdef TRISC_PACK
constexpr uint32_t MOP_LAST_OUTER_CFG_INDEX = 7;
inline uint32_t stream_pacr(uint32_t addr_mod, uint32_t last) {
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}
// ---- kw_fold.hpp's stream MOP (verbatim): rows of a sbw-wide block, W walks the block's tiles.
template <uint32_t sbw>
inline void stream_mop_config() {
    ckernel_template tmp(16, sbw, TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0), stream_pacr(ADDR_MOD_0, 0));
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010));
    tmp.set_last_inner_loop_instr(stream_pacr(ADDR_MOD_1, 0));
    tmp.set_last_outer_loop_instr(stream_pacr(ADDR_MOD_1, 0));
    tmp.program();
}
inline void stream_patch_close(bool close) {
    volatile uint32_t* mop_cfg = reinterpret_cast<volatile uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[MOP_LAST_OUTER_CFG_INDEX] = stream_pacr(ADDR_MOD_1, close ? 1 : 0);
}
template <uint32_t sbw>
inline void pack_stream(uint32_t addr, uint32_t r0, uint32_t h) {
    program_packer_destination((addr >> 4) - 1);
    for (uint32_t r = r0; r < r0 + h; ++r) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, (15 + r * sbw) & 0xF);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        for (uint32_t face = 0; face < 2; ++face) {
            const bool close = r + 1 == r0 + h && face == 1;
            if (close) {
                stream_patch_close(true);
            }
            ckernel_template::run();
            if (close) {
                stream_patch_close(false);
            }
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
            TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        }
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
    set_dst_write_addr(0);
}

// ---- 1-wide replayed streams.
constexpr uint32_t kReplayOffset = ckernel::packer::replay_buf_offset;
#define BENCH_PACR(am, last)             \
    TTI_PACR(                            \
        p_pacr::CFG_CTXT_0,              \
        p_pacr::NO_ROW_PAD_ZERO,         \
        p_pacr::DST_ACCESS_STRIDED_MODE, \
        am,                              \
        p_pacr::ADDR_CNT_CTXT_0,         \
        0,                               \
        p_pacr::TWO_INTFS_ACTIVE,        \
        0,                               \
        0,                               \
        p_pacr::NO_CTXT_CTRL,            \
        0,                               \
        last)

// narrow_untilize contig: 16 PACRs (ADDR_MOD_1: y += 1) per face pair, ADC moves between face pairs.
inline void rep_init() {
    load_replay_buf(kReplayOffset, 16, [] {
        for (uint32_t i = 0; i < 16; ++i) {
            BENCH_PACR(ADDR_MOD_1, 0);
        }
    });
}
inline void pack_rep(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    for (uint32_t t = t0; t < t0 + g; ++t) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        lltt::replay(kReplayOffset, 16);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        if (t + 1 < t0 + g) {
            lltt::replay(kReplayOffset, 16);
        } else {
            lltt::replay(kReplayOffset, 15);
            BENCH_PACR(ADDR_MOD_1, 1);
        }
    }
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
}

// Z-stepped: 15 PACRs y += 1, the 16th y = 0 / z += 1 (ADDR_MOD_2). Z stride is 2 faces (untilize init), so z walks
// F0|F1, F2|F3 of tile W, then tile W + 1 at z = 2, ...: a whole run of tiles is replays only.
inline void zst_init() {
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    load_replay_buf(kReplayOffset, 16, [] {
        for (uint32_t i = 0; i < 15; ++i) {
            BENCH_PACR(ADDR_MOD_1, 0);
        }
        BENCH_PACR(ADDR_MOD_2, 0);
    });
}
inline void pack_zst(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(kReplayOffset, 16);
    }
    lltt::replay(kReplayOffset, 15);
    BENCH_PACR(ADDR_MOD_2, 1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
}
#endif

}  // namespace bench

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t sections = get_compile_time_arg_val(1);
    constexpr uint32_t passes = get_compile_time_arg_val(2);
    constexpr uint32_t l1acc = get_compile_time_arg_val(3);
    constexpr uint32_t copy = get_compile_time_arg_val(4);
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_ACC = 16;
    constexpr uint32_t SECTION_BYTES = 4 * 4096;
    constexpr uint32_t sbw = variant == 0 ? 2 : (variant == 1 ? 4 : 1);

    compute_kernel_hw_startup(CB_IN, CB_ACC);
    cb_reserve_back(CB_IN, 4);
    cb_push_back(CB_IN, 4);
    cb_wait_front(CB_IN, 4);
    if constexpr (variant == 6 || variant == 10) {
        copy_init(CB_IN);
    } else {
        pack_untilize_init<sbw, sbw>(CB_IN, CB_ACC);
        if constexpr (variant <= 2) {
            PACK((bench::stream_mop_config<sbw>()));
        } else if constexpr (variant <= 5) {
            PACK((bench::rep_init()));
        } else {
            PACK((bench::zst_init()));
        }
    }
    [[maybe_unused]] uint32_t base = 0;
    PACK((base = get_local_cb_interface(get_output_id(CB_ACC)).fifo_wr_ptr << 4));
    {
        MaybeDeviceZoneScope("bench_pack");
        for (uint32_t p = 0; p < passes; ++p) {
            pack_reconfig_l1_acc(l1acc && p > 0 ? 1 : 0);
            for (uint32_t s = 0; s < sections; ++s) {
                tile_regs_acquire();
                if constexpr (copy) {
                    for (uint32_t j = 0; j < 4; ++j) {
                        copy_tile(CB_IN, j, j);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                [[maybe_unused]] const uint32_t addr = base + s * SECTION_BYTES;
                if constexpr (variant == 0) {
                    PACK((bench::pack_stream<2>(addr, 0, 2)));
                } else if constexpr (variant == 1) {
                    PACK((bench::pack_stream<4>(addr, 0, 1)));
                } else if constexpr (variant == 2) {
                    PACK((bench::pack_stream<1>(addr, 0, 4)));
                } else if constexpr (variant == 3) {
                    PACK((bench::pack_rep(addr, 0, 4)));
                } else if constexpr (variant == 4) {
                    PACK((bench::pack_rep(addr, 0, 2)));
                    PACK((bench::pack_rep(addr + 2 * 4096, 2, 2)));
                } else if constexpr (variant == 5) {
                    for (uint32_t t = 0; t < 4; ++t) {
                        PACK((bench::pack_rep(addr + t * 4096, t, 1)));
                    }
                } else if constexpr (variant == 6) {
                    for (uint32_t t = 0; t < 4; ++t) {
                        pack_tile<true>(t, CB_ACC, s * 4 + t);
                    }
                } else if constexpr (variant == 7) {
                    PACK((bench::pack_zst(addr, 0, 4)));
                } else if constexpr (variant == 8) {
                    PACK((bench::pack_zst(addr, 0, 2)));
                    PACK((bench::pack_zst(addr + 2 * 4096, 2, 2)));
                } else if constexpr (variant == 9) {
                    for (uint32_t t = 0; t < 4; ++t) {
                        PACK((bench::pack_zst(addr + t * 4096, t, 1)));
                    }
                }
                tile_regs_release();
            }
        }
    }
    if constexpr (variant != 6 && variant != 10) {
        pack_untilize_uninit(CB_ACC);
    }
    pack_reconfig_l1_acc(0);
}
