// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last, COLUMN_MAJOR_PADDED: compute. Per run of tiles inside one image row
// (one cb_in page = [32 x W] row-major block, rows (dy, c), pad rows zero), in chunks of up to CH tiles:
//   tilize the chunk (row stride = the full image row, Wt tiles)      -> cb_til
//   per batch of 8 tiles: transpose into DEST, then untilize each DEST tile on its own (block 1 x 1)
//   into cb_out: 32 x 64 B = 32/r sticks of r*32 channels.
// The output CB is the height-sharded output buffer itself; pushing 32/r sticks per tile walks it in order.
// The unpacker/packer are reconfigured once per chunk (tilize, then transpose + pack-untilize).
#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"

// tilize_block() would pass the batch width as the unpacker's row stride and remap the start tile index as if the
// block were 32 rows of batches; this block is a single row of Wt tiles, so unpack each tile with the real stride.
ALWI void tilize_tiles_of_row(uint32_t icb, uint32_t row_tiles, uint32_t first_tile, uint32_t ntiles, uint32_t ocb) {
    for (uint32_t t = 0; t < ntiles; t++) {
        UNPACK((llk_unpack_tilize(icb, first_tile + t, row_tiles)));
    }
    for (uint32_t t = 0; t < ntiles; t++) {
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, icb)));
        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, ocb, t)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t r = get_compile_time_arg_val(2);
    constexpr uint32_t ppt = get_compile_time_arg_val(3);  // pixels (sticks) per tile = 32/r
    constexpr uint32_t TB = get_compile_time_arg_val(4);   // tiles per DEST batch
    constexpr uint32_t cb_in = get_compile_time_arg_val(5);
    constexpr uint32_t cb_til = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t CH = get_compile_time_arg_val(8);   // tiles per chunk (cb_til capacity)

    const uint32_t tile0 = get_arg_val<uint32_t>(0);
    const uint32_t ntiles = get_arg_val<uint32_t>(1);
    if (ntiles == 0) {
        return;
    }
    unary_op_init_common(cb_in, cb_out);

    uint32_t g = tile0;
    const uint32_t end = tile0 + ntiles;
    while (g < end) {
        const uint32_t tr = g % Wt;
        uint32_t run = Wt - tr;
        if (run > end - g) {
            run = end - g;
        }
        cb_wait_front(cb_in, 1);
        for (uint32_t c0 = 0; c0 < run; c0 += CH) {
            const uint32_t nc = (run - c0 < CH) ? (run - c0) : CH;
            cb_reserve_back(cb_til, nc);
            tilize_init(cb_in, Wt, cb_til);
            tilize_tiles_of_row(cb_in, Wt, c0, nc, cb_til);
            tilize_uninit(cb_in, cb_til);
            cb_push_back(cb_til, nc);
            cb_wait_front(cb_til, nc);
            transpose_init(cb_til);
            pack_untilize_dest_init<1, 1>(cb_out);
            for (uint32_t t0 = 0; t0 < nc; t0 += TB) {
                const uint32_t nb = (nc - t0 < TB) ? (nc - t0) : TB;
                tile_regs_acquire();
                for (uint32_t i = 0; i < nb; i++) {
                    transpose_tile(cb_til, t0 + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < nb; i++) {
                    cb_reserve_back(cb_out, ppt);
                    pack_untilize_dest<1, 1>(cb_out, 1, 0, i);
                    cb_push_back(cb_out, ppt);
                }
                tile_regs_release();
            }
            pack_untilize_uninit(cb_out);
            cb_pop_front(cb_til, nc);
        }
        cb_pop_front(cb_in, 1);
        g += run;
    }
}
