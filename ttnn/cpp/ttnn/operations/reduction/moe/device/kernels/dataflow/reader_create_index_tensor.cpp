// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

/**
 * add a dfb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const DFBBindingToken dfb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(1);
    CoreLocalMem<volatile uint32_t> ptr(dfb.get_write_ptr());
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    dfb.push_back(1);
}

void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto K = get_arg(args::K);
    constexpr uint32_t Kt = K % 32 == 0 ? K / 32 : K / 32 + 1;

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input);

    const auto s1 = TensorAccessor(tensor::topk_mask);

    const auto s2 = TensorAccessor(tensor::expert_mask);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::input);
    DataflowBuffer dfb_topk(dfb::topk_mask);
    DataflowBuffer dfb_expert(dfb::expert_mask);

    const uint32_t tile_bytes_input = dfb_in0.get_tile_size();
    const uint32_t tile_bytes_topk = dfb_topk.get_tile_size();
    const uint32_t tile_bytes_expert = dfb_expert.get_tile_size();

    // Load all Wt expert mask tiles once, in a single burst, before the input stream loop.
    // The expert mask row is identical for every input row, so it is read once and the
    // tiles stay resident in the buffer for all Ht rows. Loading the whole row up front gives
    // the NoC a dedicated window for the expert reads before any input reads begin.
    dfb_expert.reserve_back(Wt);
    for (uint32_t j = 0; j < Wt; ++j) {
        noc.async_read(s2, dfb_expert, tile_bytes_expert, {.page_id = j}, {.offset_bytes = j * tile_bytes_expert});
    }
    noc.async_read_barrier();
    dfb_expert.push_back(Wt);

    // Stream in input tensor, buffer has four tiles as we double-buffer to continue streaming while waiting for compute
    // and we need two tiles for the bitonic sort llk We could load in an entire row of tiles at a time but that would
    // require substantially more memory (we would be double buffering four Wt sized buffers)
    uint32_t tile_id = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input: stream two tiles at a time (Wt is guaranteed to be a multiple of 2 for this kernel).
        for (uint32_t j = 0; j < Wt; j += 2) {
            dfb_in0.reserve_back(2);
            noc.async_read(s0, dfb_in0, tile_bytes_input, {.page_id = tile_id}, {.offset_bytes = 0});
            tile_id++;
            generate_index_tile(dfb::index, j);
            noc.async_read(s0, dfb_in0, tile_bytes_input, {.page_id = tile_id}, {.offset_bytes = tile_bytes_input});
            tile_id++;
            generate_index_tile(dfb::index, j + 1);
            noc.async_read_barrier();
            dfb_in0.push_back(2);
        }
    }

    // Topk mask: load a single row of Kt tiles. The compute kernel applies it via
    // add_block_bcast_rows_inplace(), which row-broadcasts this row across all Ht rows.
    uint32_t tile_id_topk = 0;
    dfb_topk.reserve_back(Kt);
    for (uint32_t j = 0; j < Kt; ++j) {
        noc.async_read(s1, dfb_topk, tile_bytes_topk, {.page_id = tile_id_topk}, {.offset_bytes = j * tile_bytes_topk});
        tile_id_topk++;
    }
    noc.async_read_barrier();
    dfb_topk.push_back(Kt);
}
