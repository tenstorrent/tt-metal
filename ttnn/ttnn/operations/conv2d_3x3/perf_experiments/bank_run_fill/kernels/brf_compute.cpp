// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_run_fill bench, compute (TRISC de-interleave, variants 4 / 7). Per staged chunk (CH pixels: 8 bank slots of
// CH / 8 rows, slot s = chunk pixels s, s + 8, ...), per 256-pixel group g: slot s rows [32 g, 32 g + 32) are one
// [32, cit]-tile row-major block. Pass 1 tilizes it into cb_tmp; pass 2 pack-untilizes it into cb_slab at tile column
// s * cit of a [32, 8 cit] row-major block (row pitch 8 pixels), which is chunk pixels 256 g .. 256 g + 255 in order.
// bf16 -> fp32 DEST -> bf16 both ways: exact.
// Two standard passes because the BH non-8-bit unpack-tilize leaves DEST row-swizzled (its packer un-swizzles in
// PackMode::Tilize); pack_untilize_dest needs face-ordered DEST. A one-pass raw path would need a custom packer MOP
// (DEST rows 2i, 2i+1 -> one L1 row at pitch 8 pb); not built here.
// The tilize source is not at the CB read pointer: the UNPACK thread's fifo_rd_ptr is moved to each slot block and
// restored before the pop (ckl::tilize reads only at the read pointer with the row stride tied to the block width).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "brf_geom.hpp"

constexpr uint32_t CB_SLAB = 0;
constexpr uint32_t CB_STAGE = 1;
constexpr uint32_t CB_TMP = 2;

void kernel_main() {
    using G = brf::Geom<
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)>;
    constexpr uint32_t pb = get_compile_time_arg_val(8);
    constexpr uint32_t CH = get_compile_time_arg_val(10);
    constexpr uint32_t cit = pb / 64;
    constexpr uint32_t slot_bytes = CH / 8 * pb;
    constexpr uint32_t chunk_pages = CH * pb / 2048;
    constexpr uint32_t mode = get_compile_time_arg_val(12);  // 0: two passes; 1: tilize pass only (cost split)

    const uint32_t m_tile_start = get_arg_val<uint32_t>(0);
    const uint32_t actual = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(CB_STAGE, CB_TMP);
    brf::ChunkCursor<G, CH> c;
    c.start(m_tile_start, actual);
    while (!c.done) {
        const uint32_t groups = static_cast<uint32_t>(c.e - c.a + 255) / 256;
        cb_wait_front(CB_STAGE, chunk_pages);
        [[maybe_unused]] uint32_t base = 0;
        UNPACK((base = get_local_cb_interface(CB_STAGE).fifo_rd_ptr));
        tilize_init(CB_STAGE, cit, CB_TMP);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t s = 0; s < 8; ++s) {
                cb_reserve_back(CB_TMP, cit);
                UNPACK((get_local_cb_interface(CB_STAGE).fifo_rd_ptr = base + ((s * slot_bytes + 32 * g * pb) >> 4)));
                tilize_block(CB_STAGE, cit, CB_TMP);
                cb_push_back(CB_TMP, cit);
            }
        }
        tilize_uninit(CB_STAGE, CB_TMP);
        UNPACK((get_local_cb_interface(CB_STAGE).fifo_rd_ptr = base));
        cb_pop_front(CB_STAGE, chunk_pages);
        if constexpr (mode == 0) {
            pack_untilize_init<cit, 8 * cit>(CB_TMP, CB_SLAB);
            for (uint32_t g = 0; g < groups; ++g) {
                cb_reserve_back(CB_SLAB, 8 * cit);
                for (uint32_t s = 0; s < 8; ++s) {
                    cb_wait_front(CB_TMP, cit);
                    pack_untilize_block<cit, 8 * cit>(CB_TMP, 1, CB_SLAB, s);
                    cb_pop_front(CB_TMP, cit);
                }
                cb_push_back(CB_SLAB, 8 * cit);
            }
            pack_untilize_uninit(CB_SLAB);
        } else {
            for (uint32_t g = 0; g < groups; ++g) {
                cb_reserve_back(CB_SLAB, 8 * cit);
                cb_wait_front(CB_TMP, 8 * cit);
                cb_pop_front(CB_TMP, 8 * cit);
                cb_push_back(CB_SLAB, 8 * cit);
            }
        }
        c.next();
    }
}
