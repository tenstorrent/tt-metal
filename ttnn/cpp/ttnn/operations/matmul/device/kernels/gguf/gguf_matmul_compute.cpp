// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the GGUF matmul: a decoded weight tile goes from DEST to the matmul and never to L1
// (api/compute/experimental/gguf_decode.h, dest_srca_mm.h). Per segment of an N tile one full-sync DEST acquire holds
// the Float32 accumulator (tile 0) and two sets of (decoded tile, page, decoder scratch), tiles 1-3 and 4-6,
// alternating per K tile. The pack thread decodes K tile k + 1 while the math thread multiplies tile k
// (gguf_decode_overlap_*); the unpacker copies the next page into DEST meanwhile. The input arrives in chunks
// (gguf_matmul_writer.cpp) and each K tile is waited for when first reached. Compile args: format id, K tiles, input
// tiles per chunk, input rows the matmul computes (8, 16 or 32). Runtime args: segments, then per segment its first and
// end K tile and its role (gguf_matmul_program_factory.cpp Role); a finish has the count of other cores' partials in
// place of its first K tile.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/ckernel_gguf_page.h"

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t whole = 0, finish = 3;  // Role
    constexpr uint32_t fmt = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t in0_rows = get_compile_time_arg_val(3);
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    constexpr bool group_step = fmt == gguf::Q4_K || fmt == gguf::Q5_K;
    constexpr auto pack = GgufDecodeThread::Pack;
    constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_in0 = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_part = tt::CBIndex::c_17, cb_remote = tt::CBIndex::c_18, cb_own = tt::CBIndex::c_20;
    const uint32_t segments = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_page, cb_in0, cb_out);
    gguf_decode_init<fmt, pack>();
    dest_srca_mm_init<in0_rows>(cb_in0, cb_page);
    gguf_decode_overlap_init();

    uint32_t in0_ready = 0;
    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t k0 = get_arg_val<uint32_t>(1 + 3 * s);
        const uint32_t k1 = get_arg_val<uint32_t>(2 + 3 * s);
        const uint32_t role = get_arg_val<uint32_t>(3 + 3 * s);
        if (role == finish) {
            // This core's partial into DEST tile 0, the k0 others' into 1 .. k0, added in fp32. Always the last
            // segment.
            const uint32_t parts = k0;
            dest_srca_mm_uninit();
            copy_init(cb_own);
            tile_regs_acquire();
            cb_wait_front(cb_own, 1);
            copy_tile(cb_own, 0, 0);
            cb_wait_front(cb_remote, parts);
            for (uint32_t p = 0; p < parts; p++) {
                copy_tile(cb_remote, p, 1 + p);
            }
            add_binary_tile_init();
            for (uint32_t p = 0; p < parts; p++) {
                add_binary_tile(0, 1 + p, 0);
            }
            cb_pop_front(cb_own, 1);
            cb_pop_front(cb_remote, parts);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            continue;
        }

        tile_regs_acquire();
        gguf_decode_overlap_acquire();
        cb_wait_front(cb_page, 1);
        dest_srca_mm_page_to_dest(cb_page, 0, 2);
        gguf_decode_overlap_page_ready();
        uint32_t set = 0;
        for (uint32_t kt = k0; kt < k1; kt++) {
            const uint32_t t = kt % L.group_tiles;
            const uint32_t out = set ? 4 : 1;
            const uint32_t next_page = set ? 2 : 5;
            const bool next = kt + 1 < k1;
            if (next) {
                dest_srca_mm_page_post(next_page);
            }
            gguf_decode_overlap_wait_page();
            if (kt == k0) {
                gguf_zero_tile<pack>(0);
            }
            // The group step for each DEST set: k0 starts a group.
            if (group_step && t < 2) {
                gguf_decode_group<fmt, pack>(out);
            }
            gguf_decode_tile<fmt, pack>(out, t);
            gguf_decode_overlap_tile_done();
            if (kt >= in0_ready) {
                in0_ready = (kt / chunk + 1) * chunk < k_tiles ? (kt / chunk + 1) * chunk : k_tiles;
                cb_wait_front(cb_in0, in0_ready);
            }
            dest_srca_mm_unpack(cb_in0, kt);
            if (next) {
                if ((kt + 1) % L.page_tiles == 0) {
                    cb_pop_front(cb_page, 1);
                    cb_wait_front(cb_page, 1);
                }
                dest_srca_mm_page_unpack(cb_page, 0);
                dest_srca_mm_page_wait(cb_page, next_page);
                gguf_decode_overlap_page_ready();
            }
            gguf_decode_overlap_wait_tile();
            dest_srca_mm_math<in0_rows>(out, 0);
            gguf_decode_overlap_matmul_done();
            set ^= 1;
        }
        cb_pop_front(cb_page, 1);
        tile_regs_commit();
        const uint32_t cb = role == whole ? cb_out : cb_part;
        cb_reserve_back(cb, 1);
        tile_regs_wait();
        if (role != whole) {
            pack_reconfig_data_format(cb_part);
        }
        pack_tile(0, cb);
        if (role != whole) {
            pack_reconfig_data_format(cb_out);
        }
        tile_regs_release();
        cb_push_back(cb, 1);
    }
    dest_srca_mm_uninit();
}
