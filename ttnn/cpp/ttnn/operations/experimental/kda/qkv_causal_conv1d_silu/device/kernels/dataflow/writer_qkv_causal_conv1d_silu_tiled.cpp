// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer of the tiled qkv_causal_conv1d_silu path (design.md section 6.5). Per step it writes the
// B output tiles of one column block at tile-row mt to q, k or v, one flush per step. Each tile is
// routed on its own, so a block may straddle the q/k/v split.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t block_tiles, uint32_t Mt, uint32_t Qt, uint32_t Kt, uint32_t Vt>
TT_KERNEL void writer(uint32_t step_start, uint32_t step_count) {
    constexpr uint32_t B = block_tiles;
    const auto q = TensorAccessor(tensor::q);
    const auto k = TensorAccessor(tensor::k);
    const auto v = TensorAccessor(tensor::v);
    DataflowBuffer out(dfb::out);
    Noc noc;

    const uint32_t tile_bytes = out.get_entry_size();
    const uint32_t step_end = step_start + step_count;
    for (uint32_t step = step_start; step < step_end; ++step) {
        const uint32_t mt = step % Mt;
        const uint32_t ct0 = (step / Mt) * B;
        out.wait_front(B);
        for (uint32_t i = 0; i < B; ++i) {
            const uint32_t ct = ct0 + i;
            const DataflowBufferArgs src{.offset_bytes = i * tile_bytes};
            if (ct < Qt) {
                noc.async_write(out, q, tile_bytes, src, {.page_id = mt * Qt + ct});
            } else if (ct < Qt + Kt) {
                noc.async_write(out, k, tile_bytes, src, {.page_id = mt * Kt + (ct - Qt)});
            } else {
                noc.async_write(out, v, tile_bytes, src, {.page_id = mt * Vt + (ct - Qt - Kt)});
            }
        }
        // The tiles have left L1, so compute may refill these entries.
        noc.async_writes_flushed();
        out.pop_front(B);
    }
    noc.async_write_barrier();
}
