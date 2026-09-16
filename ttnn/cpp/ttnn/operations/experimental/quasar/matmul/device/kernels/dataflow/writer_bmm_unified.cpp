// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul writer: stage A of the Quasar-native matmul (GH#41910).
//
// One writer per cluster. The compute kernel packs each output block one DST subblock at a time,
// subblocks in row-major order over the block and tiles in row-major order within a subblock. This
// writer mirrors that order, mapping every tile back to its (row, col) in the output and writing it
// by page id through the tensor accessor. Tiles past Mt / Nt (edge blocks) are popped but not written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t block_start = get_arg(args::block_start);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t Mt = get_arg(args::Mt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t num_block_cols = get_arg(args::num_block_cols);

    constexpr uint32_t MtNt = Mt * Nt;
    constexpr uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    constexpr uint32_t num_subblocks_h = per_core_M / out_subblock_h;
    constexpr uint32_t num_subblocks_w = per_core_N / out_subblock_w;

    const auto s = TensorAccessor(tensor::out);
    Noc noc;
    DataflowBuffer cb_out(dfb::cb_out);
    const uint32_t tile_bytes = cb_out.get_entry_size();

    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t out_batch_tile = b * MtNt;
        for (uint32_t blk = block_start; blk < block_start + num_blocks; ++blk) {
            const uint32_t m0 = (blk / num_block_cols) * per_core_M;
            const uint32_t n0 = (blk % num_block_cols) * per_core_N;
            for (uint32_t sbh = 0; sbh < num_subblocks_h; ++sbh) {
                for (uint32_t sbw = 0; sbw < num_subblocks_w; ++sbw) {
                    cb_out.wait_front(out_subblock_num_tiles);
                    uint32_t offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; ++h) {
                        const uint32_t m = m0 + sbh * out_subblock_h + h;
                        for (uint32_t w = 0; w < out_subblock_w; ++w, offset += tile_bytes) {
                            const uint32_t n = n0 + sbw * out_subblock_w + w;
                            if (m < Mt && n < Nt) {
                                noc.async_write(
                                    cb_out,
                                    s,
                                    tile_bytes,
                                    {.offset_bytes = offset},
                                    {.page_id = out_batch_tile + m * Nt + n});
                            }
                        }
                    }
                    noc.async_write_barrier();
                    cb_out.pop_front(out_subblock_num_tiles);
                }
            }
        }
    }
}
