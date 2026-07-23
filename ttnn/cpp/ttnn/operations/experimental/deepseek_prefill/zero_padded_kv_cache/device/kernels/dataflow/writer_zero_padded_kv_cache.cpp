// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

// Writes this chip's share of the pad window back to the cache: the masked partial tile (from the
// compute out CB) goes back to the boundary seq-tile, and every fully-pad seq-tile is zeroed in
// place. The owned tiles are contiguous-local, so the full tiles form a contiguous page range.
void kernel_main() {
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t zero_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(2);
    constexpr auto cache_args = TensorAccessorArgs<3>();

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);

    const ZeroPadChipWork w = zero_pad_compute_chip_work();
    if (w.count == 0) {
        return;  // nothing on this chip
    }

    const auto s = TensorAccessor(cache_args, cache_addr, cache_tile_bytes);
    const uint32_t base_page = w.batch_page_base + w.base_local_tile * w.Wt;

    Noc noc;

    uint32_t first_full_tile = 0;
    if (w.first_partial) {
        // Write the masked partial seq-tile (Wt width-tiles) back to the boundary tile.
        CircularBuffer out(out_cb);
        out.wait_front(w.Wt);
        for (uint32_t i = 0; i < w.Wt; ++i) {
            noc.async_write(
                out, s, cache_tile_bytes, {.offset_bytes = i * cache_tile_bytes}, {.page_id = base_page + i});
        }
        noc.async_write_barrier();
        out.pop_front(w.Wt);
        first_full_tile = 1;
    }

    if (first_full_tile >= w.count) {
        return;  // only a partial tile on this chip, no full pad tiles
    }

    // Zero a one-tile L1 scratch, then stream it to every full pad page. The scratch is never
    // push_back'd, so its read pointer equals the just-zeroed write pointer for the DRAM overload.
    CircularBuffer zero(zero_cb);
    zero.reserve_back(1);
    noc.async_write_zeros(zero, cache_tile_bytes);
    noc.write_zeros_l1_barrier();

    const uint32_t full_start_page = base_page + first_full_tile * w.Wt;
    const uint32_t full_end_page = base_page + w.count * w.Wt;
    for (uint32_t page = full_start_page; page < full_end_page; ++page) {
        noc.async_write_zeros(s, cache_tile_bytes, {.page_id = page}, zero);
    }
    noc.write_zeros_dram_barrier();
}
