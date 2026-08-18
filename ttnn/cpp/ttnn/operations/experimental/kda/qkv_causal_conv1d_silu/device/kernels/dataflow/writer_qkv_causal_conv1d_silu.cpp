// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Qt, uint32_t Kt, uint32_t Vt, uint32_t block_ct, uint32_t num_blocks>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto q = TensorAccessor(tensor::q);
    const auto k = TensorAccessor(tensor::k);
    const auto v = TensorAccessor(tensor::v);
    DataflowBuffer output(dfb::output);
    Noc noc;

    const uint32_t tile_bytes = output.get_entry_size();
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;
        output.wait_front(block_ct);
        for (uint32_t local_ct = 0; local_ct < block_ct; ++local_ct) {
            const uint32_t ct = ct_start + local_ct;
            if (ct < Qt) {
                noc.async_write(
                    output, q, tile_bytes, {.offset_bytes = local_ct * tile_bytes}, {.page_id = mt * Qt + ct});
            } else if (ct < Qt + Kt) {
                const uint32_t kt = ct - Qt;
                noc.async_write(
                    output, k, tile_bytes, {.offset_bytes = local_ct * tile_bytes}, {.page_id = mt * Kt + kt});
            } else {
                const uint32_t vt = ct - Qt - Kt;
                noc.async_write(
                    output, v, tile_bytes, {.offset_bytes = local_ct * tile_bytes}, {.page_id = mt * Vt + vt});
            }
        }
        noc.async_write_barrier();
        output.pop_front(block_ct);
    }
}
