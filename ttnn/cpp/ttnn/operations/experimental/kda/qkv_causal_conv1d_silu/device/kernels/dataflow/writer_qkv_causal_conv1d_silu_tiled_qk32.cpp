// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer of the tiled qkv_causal_conv1d_silu path with fused_qk_l2_norm (fp32 q/k outputs). The same as
// writer_qkv_causal_conv1d_silu_tiled.cpp, except that the q/k column blocks (ct0 < Qt + Kt) come from the fp32
// `out32` DFB (4 KB tiles) and go to the fp32 q/k tensors; the v blocks come from the bf16 `out` DFB. With
// fused_qk_l2_norm, B = 4 and Qt, Kt are multiples of 4, so a block never straddles the q/k/v boundaries.

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
    DataflowBuffer out32(dfb::out32);
    Noc noc;

    const uint32_t tile_bytes = out.get_entry_size();
    const uint32_t tile_bytes32 = out32.get_entry_size();
    const uint32_t step_end = step_start + step_count;
    for (uint32_t step = step_start; step < step_end; ++step) {
        const uint32_t mt = step % Mt;
        const uint32_t ct0 = (step / Mt) * B;
        if (ct0 < Qt + Kt) {
            out32.wait_front(B);
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t ct = ct0 + i;
                const DataflowBufferArgs src{.offset_bytes = i * tile_bytes32};
                if (ct < Qt) {
                    noc.async_write(out32, q, tile_bytes32, src, {.page_id = mt * Qt + ct});
                } else {
                    noc.async_write(out32, k, tile_bytes32, src, {.page_id = mt * Kt + (ct - Qt)});
                }
            }
            noc.async_writes_flushed();
            out32.pop_front(B);
        } else {
            out.wait_front(B);
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t ct = ct0 + i;
                const DataflowBufferArgs src{.offset_bytes = i * tile_bytes};
                noc.async_write(out, v, tile_bytes, src, {.page_id = mt * Vt + (ct - Qt - Kt)});
            }
            noc.async_writes_flushed();
            out.pop_front(B);
        }
    }
    noc.async_write_barrier();
}
