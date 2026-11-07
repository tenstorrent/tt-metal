// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the GGUF prefill matmul (gguf_matmul_prefill_program_factory.cpp): per M chunk, per K block, the
// core's kb x nc weight tiles are decoded once into c_4 (Float32, [kb][nc] tiles), then multiplied against all mc rows
// of the input (c_1: units of ur rows x kb tiles, [ur][kb]) in sh x sw subblocks. Partials accumulate in L1 (c_24,
// Float32, packer L1 accumulation); in the last K block each unit of rows goes to c_16 as bf16 once its sums are
// complete. Decode: per tile, the page is copied into DEST tile 1 and decoded into tile 0 on the math thread, then
// packed. K split (krole): band 1 packs its sums to c_17 for the writer to send, band 0 adds the partner's (c_18).
// Compile args: format id, mc, kb, K blocks, M chunks, ur (sh divides it). Runtime args: N tiles of this core (0:
// nothing to do), N tiles the circular buffers are sized for (c_4 and c_24 advance by whole blocks of that width, so
// that c_24's L1 accumulation lands on the same tiles), sh, sw, krole (0: no K split, 1: send sums, 2: add them).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/ckernel_gguf_page.h"

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t fmt = get_compile_time_arg_val(0);
    constexpr uint32_t mc = get_compile_time_arg_val(1);
    constexpr uint32_t kb = get_compile_time_arg_val(2);
    constexpr uint32_t kblocks = get_compile_time_arg_val(3);
    constexpr uint32_t mchunks = get_compile_time_arg_val(4);
    constexpr uint32_t ur = get_compile_time_arg_val(5);  // input rows per unit (c_1 holds units of ur x kb tiles)
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    constexpr bool group_step = fmt == gguf::Q4_K || fmt == gguf::Q5_K || fmt == gguf::Q3_K || fmt == gguf::Q2_K;
    constexpr uint32_t block_groups = kb / L.group_tiles;
    constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_in0 = tt::CBIndex::c_1, cb_w = tt::CBIndex::c_4;
    constexpr uint32_t cb_out = tt::CBIndex::c_16, cb_acc = tt::CBIndex::c_24;
    constexpr uint32_t cb_send = tt::CBIndex::c_17, cb_recv = tt::CBIndex::c_18;
    const uint32_t nc = get_arg_val<uint32_t>(0);
    const uint32_t nca = get_arg_val<uint32_t>(1);
    const uint32_t sh = get_arg_val<uint32_t>(2);
    const uint32_t sw = get_arg_val<uint32_t>(3);
    const uint32_t krole = get_arg_val<uint32_t>(4);  // K split: 1 sends its sums (c_17), 2 adds the partner's (c_18)
    if (nc == 0) {
        return;
    }
    const uint32_t sub = sh * sw, w_tiles = kb * nca, acc_tiles = mc * nc, acc_ring = mc * nca;

    compute_kernel_hw_startup(cb_page, cb_in0, cb_w);
    uint32_t srca = cb_page;  // the CB SrcA is configured for
    auto srca_to = [&](uint32_t cb) {
        if (srca != cb) {
            reconfig_data_format_srca(srca, cb);
            srca = cb;
        }
    };

    for (uint32_t chunk = 0; chunk < mchunks; chunk++) {
        for (uint32_t blk = 0; blk < kblocks; blk++) {
            // Decode the K block's weight tiles, in record order (per group, per N tile, per tile of the group).
            srca_to(cb_page);
            gguf_decode_init<fmt>();
            dest_srca_mm_init<32>(cb_in0, cb_page);
            pack_reconfig_data_format(cb_w);
            pack_reconfig_l1_acc(0);
            cb_reserve_back(cb_w, w_tiles);
            // Formats with a group step keep DEST across releases inside the block (half-sync: tile t of a record is
            // in the half of t % 2), so the step runs once per record and half, not per tile. The release after the
            // last tile of each half clears it for the matmul.
            uint32_t left = kb * nc;
            for (uint32_t g = 0; g < block_groups; g++) {
                for (uint32_t j = 0; j < nc; j++) {
                    for (uint32_t t = 0; t < L.group_tiles; t++, left--) {
                        if (t % L.page_tiles == 0) {
                            cb_wait_front(cb_page, 1);
                        }
                        tile_regs_acquire();
                        dest_srca_mm_page_to_dest(cb_page, 0, 1);
                        if (group_step && t < 2) {
                            gguf_decode_group<fmt>(0);
                        }
                        gguf_decode_tile<fmt>(0, t);
                        tile_regs_commit();
                        if ((t + 1) % L.page_tiles == 0) {
                            cb_pop_front(cb_page, 1);
                        }
                        tile_regs_wait();
                        pack_tile<true>(0, cb_w, (g * L.group_tiles + t) * nca + j);
                        if (group_step && left > 2) {
                            PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>()));
                        } else {
                            tile_regs_release();
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
                            // Own sums + the partner's, in fp32 on the SFPU, two tiles at a time.
                            cb_wait_front(cb_recv, ur * nca);
                            for (uint32_t i = 0; i < ur * nc; i += 2) {
                                const uint32_t n2 = ur * nc - i < 2 ? 1 : 2;
                                tile_regs_acquire();
                                copy_init(cb_acc);
                                for (uint32_t t = 0; t < n2; t++) {
                                    copy_tile(cb_acc, i + t, t);
                                    copy_tile(cb_recv, i + t, 2 + t);
                                }
                                add_binary_tile_init();
                                for (uint32_t t = 0; t < n2; t++) {
                                    add_binary_tile(t, 2 + t, t);
                                }
                                tile_regs_commit();
                                cb_reserve_back(cb_out, n2);
                                tile_regs_wait();
                                for (uint32_t t = 0; t < n2; t++) {
                                    pack_tile(t, cb_out);
                                }
                                tile_regs_release();
                                cb_push_back(cb_out, n2);
                            }
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
