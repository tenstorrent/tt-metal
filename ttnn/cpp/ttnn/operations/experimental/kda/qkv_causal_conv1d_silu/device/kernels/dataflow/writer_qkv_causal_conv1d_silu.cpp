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
    const auto state_source = TensorAccessor(tensor::state_source);
    const auto state = TensorAccessor(tensor::state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer state_copy(dfb::state_copy);
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

        if (mt == 0) {
            constexpr uint32_t history_rows = 3;
            constexpr uint32_t shard_channels = 64;
            constexpr uint32_t row_bytes = shard_channels * sizeof(uint16_t);
            constexpr uint32_t state_page_bytes = history_rows * row_bytes;
            constexpr uint32_t pages_per_block = block_ct * 32 / shard_channels;
            const uint32_t block = work % num_blocks;
            for (uint32_t local_page = 0; local_page < pages_per_block; ++local_page) {
                const uint32_t page = block * pages_per_block + local_page;
                const uint32_t channel_offset = page * row_bytes;
                state_copy.reserve_back(1);
                for (uint32_t row = 0; row < history_rows; ++row) {
                    noc.async_read(
                        state_source,
                        state_copy,
                        row_bytes,
                        {.page_id = row, .offset_bytes = channel_offset},
                        {.offset_bytes = row * row_bytes});
                }
                noc.async_read_barrier();
                state_copy.push_back(1);
                state_copy.wait_front(1);
                noc.async_write(state_copy, state, state_page_bytes, {}, {.page_id = page});
                noc.async_write_barrier();
                state_copy.pop_front(1);
            }
        }
    }
}
