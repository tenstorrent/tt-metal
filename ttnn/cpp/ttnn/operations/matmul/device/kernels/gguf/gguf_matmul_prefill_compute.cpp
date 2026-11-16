// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the GGUF prefill matmul (gguf_matmul_prefill_program_factory.cpp): per M chunk, per K block, the
// core's kb x nc weight tiles are decoded once into c_4 (Float32, [kb][nc] tiles), then multiplied against all mc rows
// of the input (c_1: units of ur rows x kb tiles, [ur][kb]) in sh x sw subblocks. Partials accumulate in L1 (c_24,
// Float32, packer L1 accumulation); in the last K block each unit of rows goes to c_16 as bf16 once its sums are
// complete. Decode on the math thread: GGUF_PF_PIPE 1, decode_block (the next page's copy beside each decode);
// GGUF_PF_PIPE 0, per tile the page is copied into DEST tile 1 and decoded into tile 0 (block-float codes: tiles 2 and
// 1), then packed. K split (krole): band 1 packs its sums to c_17 for the writer to send, band 0 adds the partners'
// (c_18); both in units of ur rows x the planned N tiles, the rows of each N tile together (subblocks of ur x 1).
// Compile args: format id, mc, kb, M chunks, ur (sh divides it), N passes (the core's N tiles in passes of up to the
// width below; a pass without any only consumes the input), c_16's pages. Runtime args: N
// tiles of this core (0: nothing to do), N tiles the circular buffers are sized for (c_4 and c_24 advance by whole
// blocks of that width, so that c_24's L1 accumulation lands on the same tiles), sh, sw, krole (0: no K split, 1: send
// sums, 2: add them), K blocks.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/ckernel_gguf_page.h"

namespace {

using namespace ckernel;

// Decodes a K block's weight tiles (record order: per group, per N tile, per tile of the group) into c_4 in a
// full-sync DEST section: the unpacker copies the page of tile i + 1 while the math thread's SFPU decodes tile i and
// the packer packs tile i - 1. Two sets of (decoded tile, page, scratch), DEST tiles 1-3 and 4-6; tile i takes set
// i % 2, so a record's tiles alternate between the two sets as its group step needs. Tensix semaphores: FPU_SFPU,
// math -> pack: a decoded tile is complete; PACK_DONE, pack -> math: a set's decoded tile has been packed. The
// half-sync DEST of the matmul is set up again at the end. A decoder whose phase 2 may not run beside a copy
// (gguf_decode.h) gets the next page copied after it instead.
template <uint32_t fmt>
constexpr bool decode_block_fits() {
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    return !gguf::bfp_codes<fmt> && L.page_tiles == 1 && (L.group_tiles == 1 || L.group_tiles % 2 == 0);
}

template <uint32_t fmt>
ALWI void decode_block(const uint32_t nc, const uint32_t nca, const uint32_t groups) {
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    constexpr bool group_step = gguf::group_scales<fmt>;
    constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_w = tt::CBIndex::c_4;
#ifdef TRISC_MATH
    constexpr bool beside = !llk_math_gguf_decode_body_unarbitrated<fmt>();
#else
    constexpr bool beside = true;
#endif
    if constexpr (!decode_block_fits<fmt>()) {
        return;
    }
    auto out_of = [](uint32_t set) -> uint32_t { return set ? 4 : 1; };
    const uint32_t tiles = groups * nc * L.group_tiles;
    MATH((_llk_math_pack_sync_init_<DstSync::SyncFull, DST_ACCUM_MODE>()));
    PACK((_llk_pack_dest_init_<DstSync::SyncFull, DST_ACCUM_MODE>()));
    MATH((_llk_math_wait_for_dest_available_<DstSync::SyncFull>()));
    gguf_decode_overlap_acquire();
    MATH((t6_semaphore_init(semaphore::FPU_SFPU, 0, 2)));
    MATH((t6_semaphore_init(semaphore::PACK_DONE, 2, 2)));
    dest_srca_mm_page_post(out_of(0) + 1);
    for (uint32_t i = 0; i < tiles; i++) {
        const uint32_t set = i & 1, t = i % L.group_tiles, rec = i / L.group_tiles;
        cb_wait_front(cb_page, 1);
        dest_srca_mm_page_unpack(cb_page, 0);
        cb_pop_front(cb_page, 1);
        dest_srca_mm_page_wait(cb_page, out_of(set) + 1);
        if (beside && i + 1 < tiles) {
            dest_srca_mm_page_post(out_of(set ^ 1) + 1);
        }
        MATH(({
            TTI_SEMWAIT(
                p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
                semaphore::t6_sem(semaphore::PACK_DONE),
                p_stall::STALL_ON_ZERO);
            t6_semaphore_get<p_stall::NONE>(semaphore::PACK_DONE);
        }));
        if (group_step && t < 2) {
            gguf_decode_group<fmt>(out_of(set));
        }
        gguf_decode_tile<fmt, GgufDecodeThread::Math, true>(out_of(set), t);
        if (!beside && i + 1 < tiles) {
            dest_srca_mm_page_post(out_of(set ^ 1) + 1);
        }
        MATH(({ t6_semaphore_post<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::FPU_SFPU); }));
        PACK(({
            TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_SYNC,
                semaphore::t6_sem(semaphore::FPU_SFPU),
                p_stall::STALL_ON_ZERO);
            t6_semaphore_get<p_stall::NONE>(semaphore::FPU_SFPU);
        }));
        pack_tile<true>(out_of(set), cb_w, ((rec / nc) * L.group_tiles + t) * nca + rec % nc);
        PACK((t6_semaphore_post<p_stall::PACK>(semaphore::PACK_DONE)));
    }
    // Both sets packed (PACK_DONE back at 0, as the decode matmul expects it), then the section ends.
    MATH(({
        for (uint32_t k = 0; k < 2; k++) {
            TTI_SEMWAIT(
                p_stall::STALL_MATH | p_stall::STALL_SYNC,
                semaphore::t6_sem(semaphore::PACK_DONE),
                p_stall::STALL_ON_ZERO);
            t6_semaphore_get<p_stall::NONE>(semaphore::PACK_DONE);
        }
    }));
    MATH((_llk_math_dest_section_done_<DstSync::SyncFull, DST_ACCUM_MODE>()));
    PACK((_llk_packer_wait_for_math_done_()));
    PACK((_llk_pack_dest_section_done_<DstSync::SyncFull, DST_ACCUM_MODE>()));
    MATH((_llk_math_pack_sync_init_<DstSync::SyncHalf, DST_ACCUM_MODE>()));
    PACK((_llk_pack_dest_init_<DstSync::SyncHalf, DST_ACCUM_MODE>()));
}

}  // namespace

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t fmt = get_compile_time_arg_val(0);
    constexpr uint32_t mc = get_compile_time_arg_val(1);
    constexpr uint32_t kb = get_compile_time_arg_val(2);
    constexpr uint32_t mchunks = get_compile_time_arg_val(3);
    constexpr uint32_t ur = get_compile_time_arg_val(4);  // input rows per unit (c_1 holds units of ur x kb tiles)
    constexpr uint32_t passes = get_compile_time_arg_val(5);
    constexpr uint32_t out_pages = get_compile_time_arg_val(6);
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    constexpr bool group_step = gguf::group_scales<fmt>;
    constexpr uint32_t block_groups = kb / L.group_tiles;
    // Decoded tile 0, its page in tile 1. Block-float codes: decoded tile 1, page 2, the codes in the 16-bit
    // rows that tile 1's rows address (in tile 0, dest_srca_mm_page_unpack_codes).
    constexpr bool bfp = gguf::bfp_codes<fmt>;
    constexpr uint32_t out = bfp ? 1 : 0;
    // Without decode_block each page is copied before its decode starts, so no unpack to DEST runs beside one:
    // the faster decoders.
    constexpr auto thr = GgufDecodeThread::Math;
    constexpr bool beside_unpack = false;
    // The pipelined decode (decode_block) where the plan chose it and the format's pages allow it.
    constexpr bool pipe = GGUF_PF_PIPE && decode_block_fits<fmt>();
    constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_in0 = tt::CBIndex::c_1, cb_w = tt::CBIndex::c_4;
    constexpr uint32_t cb_out = tt::CBIndex::c_16, cb_acc = tt::CBIndex::c_24;
    constexpr uint32_t cb_send = tt::CBIndex::c_17, cb_recv = tt::CBIndex::c_18;
    const uint32_t core_nc = get_arg_val<uint32_t>(0);
    const uint32_t nca = get_arg_val<uint32_t>(1);
    const uint32_t sh = get_arg_val<uint32_t>(2);
    const uint32_t sw = get_arg_val<uint32_t>(3);
    const uint32_t krole = get_arg_val<uint32_t>(4);  // K split: 1 sends its sums (c_17), 2 adds the partner's (c_18)
    const uint32_t kblocks = get_arg_val<uint32_t>(5);
    if (core_nc == 0) {
        return;
    }
    const uint32_t sub = sh * sw, w_tiles = kb * nca, acc_ring = mc * nca;
    uint32_t out_page = 0;  // krole 2: c_16's write page

    compute_kernel_hw_startup(cb_page, cb_in0, cb_w);
    uint32_t srca = cb_page;  // the CB SrcA is configured for
    auto srca_to = [&](uint32_t cb) {
        if (srca != cb) {
            reconfig_data_format_srca(srca, cb);
            srca = cb;
        }
    };

    for (uint32_t chunk = 0; chunk < passes * mchunks; chunk++) {
        const uint32_t pass = passes == 1 ? 0 : chunk / mchunks;
        const uint32_t nc = passes == 1            ? core_nc
                            : core_nc > pass * nca ? (core_nc - pass * nca < nca ? core_nc - pass * nca : nca)
                                                   : 0;
        const uint32_t acc_tiles = mc * nc;
        if (passes > 1 && nc == 0) {
            for (uint32_t unit = 0; unit < kblocks * (mc / ur); unit++) {
                cb_wait_front(cb_in0, ur * kb);
                cb_pop_front(cb_in0, ur * kb);
            }
            continue;
        }
        for (uint32_t blk = 0; blk < kblocks; blk++) {
            // Decode the K block's weight tiles, in record order (per group, per N tile, per tile of the group).
            srca_to(cb_page);
            if constexpr (pipe) {
                gguf_decode_init<fmt, thr, true>();
                dest_srca_mm_init<32>(cb_in0, cb_page);
                pack_reconfig_data_format(cb_w);
                pack_reconfig_l1_acc(0);
                cb_reserve_back(cb_w, w_tiles);
                decode_block<fmt>(nc, nca, block_groups);
            } else {
                gguf_decode_init<fmt, thr, beside_unpack>();
                dest_srca_mm_init<32>(cb_in0, cb_page);
                pack_reconfig_data_format(cb_w);
                pack_reconfig_l1_acc(0);
                cb_reserve_back(cb_w, w_tiles);
                // Formats with a group step keep DEST across releases inside the block (half-sync: tile t of a record
                // is in the half of t % 2), so the step runs once per record and half, not per tile. The release after
                // the last tile of each half clears it for the matmul.
                uint32_t left = kb * nc;
                for (uint32_t g = 0; g < block_groups; g++) {
                    for (uint32_t j = 0; j < nc; j++) {
                        for (uint32_t t = 0; t < L.group_tiles; t++, left--) {
                            if (t % L.page_tiles == 0) {
                                cb_wait_front(cb_page, 1);
                            }
                            tile_regs_acquire();
                            if constexpr (bfp) {
                                dest_srca_mm_page_to_dest_codes(
                                    cb_page,
                                    0,
                                    out + 1,
                                    out,
                                    gguf::BFP_CODES,
                                    gguf::BFP_CODE_EXP,
                                    L.scale_offset,
                                    gguf::bfp_scale_words<fmt>());
                            } else {
                                dest_srca_mm_page_to_dest(cb_page, 0, out + 1);
                            }
                            if (group_step && t < 2) {
                                gguf_decode_group<fmt>(out);
                            }
                            gguf_decode_tile<fmt, thr, beside_unpack>(out, t);
                            tile_regs_commit();
                            if ((t + 1) % L.page_tiles == 0) {
                                cb_pop_front(cb_page, 1);
                            }
                            tile_regs_wait();
                            pack_tile<true>(out, cb_w, (g * L.group_tiles + t) * nca + j);
                            if (group_step && left > 2) {
                                PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>()));
                            } else {
                                tile_regs_release();
                            }
                        }
                    }
                }
            }
            cb_push_back(cb_w, w_tiles);
            dest_srca_mm_uninit();

            // The matmul of the block: partials into c_24 (L1 accumulation after the first K block).
            srca_to(cb_w);
            matmul_block_init(cb_in0, cb_w, 0, sw, sh, kb);
            pack_reconfig_data_format(cb_acc);
            pack_reconfig_l1_acc(blk > 0 ? 1 : 0);
            cb_wait_front(cb_w, w_tiles);
            for (uint32_t r = 0; r < mc; r += sh) {
                if (r % ur == 0) {
                    cb_wait_front(cb_in0, ur * kb);
                }
                for (uint32_t c = 0; c < nc; c += sw) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < kb; k++) {
                        matmul_block(cb_in0, cb_w, (r % ur) * kb + k, k * nca + c, 0, 0, sw, sh, kb);
                    }
                    tile_regs_commit();
                    cb_reserve_back(cb_acc, sub);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < sub; i++) {
                        pack_tile(i, cb_acc);
                    }
                    tile_regs_release();
                    cb_push_back(cb_acc, sub);
                }
                if ((r + sh) % ur == 0) {
                    cb_pop_front(cb_in0, ur * kb);
                    if (blk + 1 == kblocks) {
                        // Last K block: the unit's sums to bf16 now, so that the output moves while the next units
                        // multiply.
                        const uint32_t to = krole == 1 ? cb_send : cb_out;
                        pack_reconfig_l1_acc(0);
                        pack_reconfig_data_format(to);
                        srca_to(cb_acc);
                        cb_wait_front(cb_acc, ur * nc);
                        if (krole == 2) {
                            // Own sums + the partner's, added by the FPU (both in fp32 L1), up to four tiles at a time;
                            // fewer where a group would cross the end of c_16.
                            cb_wait_front(cb_recv, ur * nca);
                            reconfig_data_format_srcb(cb_in0, cb_recv);
                            add_init(cb_acc, cb_recv);
                            for (uint32_t i = 0, n4 = 0; i < ur * nc; i += n4) {
                                n4 = ur * nc - i < 4 ? ur * nc - i : 4;
                                n4 = out_pages - out_page < n4 ? out_pages - out_page : n4;
                                out_page = out_page + n4 == out_pages ? 0 : out_page + n4;
                                tile_regs_acquire();
                                for (uint32_t t = 0; t < n4; t++) {
                                    add_tiles(cb_acc, cb_recv, i + t, i + t, t);
                                }
                                tile_regs_commit();
                                cb_reserve_back(cb_out, n4);
                                tile_regs_wait();
                                for (uint32_t t = 0; t < n4; t++) {
                                    pack_tile(t, cb_out);
                                }
                                tile_regs_release();
                                cb_push_back(cb_out, n4);
                            }
                            reconfig_data_format_srcb(cb_recv, cb_in0);
                            cb_pop_front(cb_recv, ur * nca);
                        } else {
                            copy_init(cb_acc);
                            for (uint32_t i = 0; i < ur * nc; i += sub) {
                                tile_regs_acquire();
                                for (uint32_t t = 0; t < sub; t++) {
                                    copy_tile(cb_acc, i + t, t);
                                }
                                tile_regs_commit();
                                cb_reserve_back(to, sub);
                                tile_regs_wait();
                                for (uint32_t t = 0; t < sub; t++) {
                                    pack_tile(t, to);
                                }
                                tile_regs_release();
                                cb_push_back(to, sub);
                            }
                            if (krole == 1 && nca > nc) {
                                // Units of sums are ur x nca tiles, so that they never wrap around the rings.
                                cb_reserve_back(cb_send, ur * (nca - nc));
                                cb_push_back(cb_send, ur * (nca - nc));
                            }
                        }
                        cb_pop_front(cb_acc, ur * nc);
                        if (r + sh < mc) {
                            srca_to(cb_w);
                            matmul_block_init(cb_in0, cb_w, 0, sw, sh, kb);
                            pack_reconfig_data_format(cb_acc);
                            pack_reconfig_l1_acc(blk > 0 ? 1 : 0);
                        }
                    }
                }
            }
            cb_pop_front(cb_w, w_tiles);
            if (acc_ring > acc_tiles) {
                cb_reserve_back(cb_acc, acc_ring - acc_tiles);
                cb_push_back(cb_acc, acc_ring - acc_tiles);
            }
            // Every subblock's partial is back at the same place of c_24 (it holds exactly acc_ring tiles).
            if (blk + 1 < kblocks) {
                cb_wait_front(cb_acc, acc_ring);
                cb_pop_front(cb_acc, acc_ring);
            } else if (acc_ring > acc_tiles) {
                cb_wait_front(cb_acc, acc_ring - acc_tiles);
                cb_pop_front(cb_acc, acc_ring - acc_tiles);
            }
        }
        pack_reconfig_l1_acc(0);
    }
}
