// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"

// Sharded layernorm over an interleaved (DRAM or L1) input or output: each core owns a block_ht x block_wt block of the
// [Mt, Kt] tile grid, whose first tile is first_tile_id, stored row by row in its buffer.

// Read this core's input block into in0, which compute waits on.
template <uint32_t block_ht, uint32_t block_wt, uint32_t Kt, typename DFBToken, typename TensorToken>
inline void read_interleaved_block(
    const Noc& noc, const DFBToken& in0_token, const TensorToken& src_token, uint32_t first_tile_id) {
    DataflowBuffer dfb_in0(in0_token);
    const auto src = TensorAccessor(src_token);
    const uint32_t tile_bytes = dfb_in0.get_tile_size();
    dfb_in0.reserve_back(block_ht * block_wt);
    uint32_t offset = 0;
    for (uint32_t h = 0; h < block_ht; ++h) {
        for (uint32_t w = 0; w < block_wt; ++w) {
            noc.async_read(
                src, dfb_in0, tile_bytes, {.page_id = first_tile_id + (h * Kt) + w}, {.offset_bytes = offset});
            offset += tile_bytes;
        }
    }
    noc.async_read_barrier();
    dfb_in0.push_back(block_ht * block_wt);
}

// Write this core's output block, tile by tile as compute produces it, to its interleaved pages.
template <uint32_t block_ht, uint32_t block_wt, uint32_t Kt, typename DFBToken, typename TensorToken>
inline void write_interleaved_block(
    const Noc& noc, const DFBToken& out_token, const TensorToken& dst_token, uint32_t first_tile_id) {
    DataflowBuffer dfb_out(out_token);
    const auto dst = TensorAccessor(dst_token);
    const uint32_t tile_bytes = dfb_out.get_tile_size();
    uint32_t num_written = 0;
    for (uint32_t h = 0; h < block_ht; ++h) {
        for (uint32_t w = 0; w < block_wt; ++w) {
            dfb_out.wait_front(static_cast<uint16_t>(num_written + 1));
            noc.async_write(
                dfb_out,
                dst,
                tile_bytes,
                {.offset_bytes = num_written * tile_bytes},
                {.page_id = first_tile_id + (h * Kt) + w});
            ++num_written;
        }
    }
    noc.async_write_barrier();
    dfb_out.pop_front(static_cast<uint16_t>(num_written));
}
