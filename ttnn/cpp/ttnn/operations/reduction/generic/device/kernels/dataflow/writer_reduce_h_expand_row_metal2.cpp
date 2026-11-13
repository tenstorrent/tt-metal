// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar writer for the tiled reduce-H factories (reduction/generic and experimental/quasar/reduction).
//
// On Quasar a column reduce leaves its result in DEST as a 1x32 tile, so compute packs each result into `out` as a
// 1x32 tile: one row of 32 datums, face 0's 16 and then face 1's 16. The op's output tensor keeps 32x32 tile pages
// (row 0 holds the result, every other row is zero), so this writer builds each page in the `full_tile` scratchpad:
// zeroed once, then per tile face 0's row goes to row 0 of face 0 and face 1's row to row 0 of face 1.
// The copy is datum by datum, so the output format must not be block-float.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t datum_bytes = get_arg(args::datum_bytes);

    constexpr uint32_t face_dim = 16;
    constexpr uint32_t row_words = face_dim * datum_bytes / sizeof(uint32_t);  // one face row
    constexpr uint32_t face_words = face_dim * row_words;                      // one 16x16 face
    constexpr uint32_t tile_words = 4 * face_words;                            // one 32x32 tile
    constexpr uint32_t row_bytes = row_words * sizeof(uint32_t);
    constexpr uint32_t tile_bytes = tile_words * sizeof(uint32_t);

    Noc noc;
    DataflowBuffer out(dfb::out);
    Scratchpad<uint32_t> full_tile(scratch::full_tile);
    const auto s = TensorAccessor(tensor::output);
    const uintptr_t full_tile_addr = local_mem::detail::AddressAccess::address(full_tile.local_mem());

    // Rows 1-31 of every page are zero; only row 0 of faces 0 and 1 is rewritten per tile.
    for (uint32_t w = 0; w < tile_words; ++w) {
        full_tile[w] = 0;
    }
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // CPU stores land in this core's caches; write them back so the NoC reads them from L1.
    flush_l2_cache_range(full_tile_addr, tile_bytes);
#endif

    for (uint32_t page_id = start_id; page_id < start_id + num_pages; ++page_id) {
        out.wait_front(1);
        const uint32_t row_addr = out.get_read_ptr();
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // The packer wrote the row straight to L1, past this core's caches: drop any stale cached copy first.
        invalidate_l2_cache_range(row_addr, 2 * row_bytes);
#endif
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_addr);
        for (uint32_t w = 0; w < row_words; ++w) {
            full_tile[w] = row[w];
            full_tile[face_words + w] = row[row_words + w];
        }
        out.pop_front(1);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        flush_l2_cache_range(full_tile_addr, row_bytes);
        flush_l2_cache_range(full_tile_addr + face_words * sizeof(uint32_t), row_bytes);
#endif
        noc.async_write(full_tile, s, tile_bytes, {.offset_bytes = 0}, {.page_id = page_id});
        // The next tile reuses full_tile, so its data must have left L1 before it is overwritten.
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}
