// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

// Reads the boundary (partial) pad tile from the cache into the src CB, and builds the bf16 row-mask
// tile (rows [0,row_start) = 1.0, rows [row_start,32) = 0.0) into the mask CB. Only runs on the chip
// that owns the partial tile; chips with only full pad tiles do nothing here.
void kernel_main() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mask_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(2);
    constexpr auto cache_args = TensorAccessorArgs<3>();

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);

    const ZeroPadChipWork w = zero_pad_compute_chip_work();
    if (w.count == 0 || w.first_partial == 0) {
        return;  // this chip has no partial tile to mask
    }

    const auto s = TensorAccessor(cache_args, cache_addr, cache_tile_bytes);
    const uint32_t base_page = w.batch_page_base + w.base_local_tile * w.Wt;

    Noc noc;
    CircularBuffer src(src_cb);

    // Read the Wt width-tiles of the boundary seq-tile into the src CB.
    src.reserve_back(w.Wt);
    for (uint32_t i = 0; i < w.Wt; ++i) {
        noc.async_read(s, src, cache_tile_bytes, {.page_id = base_page + i}, {.offset_bytes = i * cache_tile_bytes});
    }
    noc.async_read_barrier();
    src.push_back(w.Wt);

    // Build the bf16 row-mask tile in L1 (face layout: 4x 16x16 faces, order TL,TR,BL,BR).
    constexpr uint16_t kBf16One = 0x3F80;  // 1.0
    CircularBuffer mask(mask_cb);
    mask.reserve_back(1);
    CoreLocalMem<uint16_t> m(mask.get_write_ptr());
    const uint32_t rs = w.row_start;
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t row_base = (face >= 2) ? 16u : 0u;  // faces 0,1 -> rows 0-15; 2,3 -> rows 16-31
        for (uint32_t fr = 0; fr < 16; ++fr) {
            const uint16_t val = ((row_base + fr) < rs) ? kBf16One : 0u;
            for (uint32_t fc = 0; fc < 16; ++fc) {
                m[face * 256 + fr * 16 + fc] = val;
            }
        }
    }
    mask.push_back(1);
}
