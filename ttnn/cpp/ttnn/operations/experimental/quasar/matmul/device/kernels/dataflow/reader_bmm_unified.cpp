// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul reader: stage A of the Quasar-native matmul (GH#41910).
//
// One reader per cluster. It walks the cluster's run of output blocks (block_start .. +num_blocks, in
// row-major block order) and for every K block pushes
//   - one in0 block: [per_core_M x in0_block_w] tiles, row-major, and
//   - one in1 block: [in0_block_w x per_core_N] tiles, row-major,
// in exactly the layout bmm_large_block_zm_fused_bias_activation_metal2.cpp consumes. Loop order
// (batch, block, K block) matches that kernel's (batch, num_blocks_h_dim, num_blocks_inner_dim).
//
// Edge blocks: rows past Mt and columns past Nt are never read. Their slots keep stale L1, which can
// only reach output tiles the writer drops (a valid output tile uses valid in0 rows and valid in1
// columns only). The padding columns of in0's last K tile are zeroed so K padding contributes 0.
//
// Both operands are addressed by page id through the tensor accessor, so interleaved, L1-sharded and
// DRAM-sharded inputs are one code path. Locality (reading a shard that already lives on this core
// without the copy) is a later stage.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    // Per-core: this cluster's contiguous run of output blocks.
    const uint32_t block_start = get_arg(args::block_start);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t Mt = get_arg(args::Mt);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t bcast_batch = get_arg(args::bcast_batch);
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_block_cols = get_arg(args::num_block_cols);
    constexpr uint32_t in0_last_ktile_w = get_arg(args::in0_last_ktile_w);

    constexpr uint32_t in0_block_num_tiles = per_core_M * in0_block_w;
    constexpr uint32_t in1_block_num_tiles = in0_block_w * per_core_N;
    constexpr uint32_t MtKt = Mt * Kt;
    constexpr uint32_t KtNt = Kt * Nt;

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;
    constexpr uint32_t cb_id_in1 = dfb::cb_in1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    const auto s0 = TensorAccessor(tensor::in0);
    const auto s1 = TensorAccessor(tensor::in1);

    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in0);
    DataflowBuffer cb_in1(dfb::cb_in1);
    // Entry size is the (DRAM-aligned) tile stride the factory sized the ring with.
    const uint32_t in0_stride = cb_in0.get_entry_size();
    const uint32_t in1_stride = cb_in1.get_entry_size();

    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t in0_batch_tile = b * MtKt;
        const uint32_t in1_batch_tile = bcast_batch ? 0 : b * KtNt;
        for (uint32_t blk = block_start; blk < block_start + num_blocks; ++blk) {
            const uint32_t m0 = (blk / num_block_cols) * per_core_M;
            const uint32_t n0 = (blk % num_block_cols) * per_core_N;
            const uint32_t valid_h = (Mt - m0 < per_core_M) ? (Mt - m0) : per_core_M;
            const uint32_t valid_w = (Nt - n0 < per_core_N) ? (Nt - n0) : per_core_N;

            for (uint32_t kb = 0; kb < num_blocks_inner_dim; ++kb) {
                const uint32_t k0 = kb * in0_block_w;

                // in0 block: rows m0.., cols k0.. (only the valid rows; invalid rows trail).
                cb_in0.reserve_back(in0_block_num_tiles);
                {
                    uint32_t offset = 0;
                    for (uint32_t h = 0; h < valid_h; ++h) {
                        uint32_t page = in0_batch_tile + (m0 + h) * Kt + k0;
                        for (uint32_t w = 0; w < in0_block_w; ++w, ++page, offset += in0_stride) {
                            noc.async_read(s0, cb_in0, in0_tile_bytes, {.page_id = page}, {.offset_bytes = offset});
                        }
                    }
                }

                // in1 block: rows k0.., cols n0.. (invalid columns are skipped but keep their slot).
                cb_in1.reserve_back(in1_block_num_tiles);
                {
                    uint32_t offset = 0;
                    for (uint32_t kk = 0; kk < in0_block_w; ++kk) {
                        uint32_t page = in1_batch_tile + (k0 + kk) * Nt + n0;
                        for (uint32_t w = 0; w < per_core_N; ++w, ++page, offset += in1_stride) {
                            if (w < valid_w) {
                                noc.async_read(s1, cb_in1, in1_tile_bytes, {.page_id = page}, {.offset_bytes = offset});
                            }
                        }
                    }
                }

                noc.async_read_barrier();

                if constexpr (in0_last_ktile_w > 0) {
                    // Zero the padding columns of the last K tile of every valid row (reads are done).
                    if (kb == num_blocks_inner_dim - 1) {
                        constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                        const uint32_t base = cb_in0.get_write_ptr() + (in0_block_w - 1) * in0_stride;
                        for (uint32_t h = 0; h < valid_h; ++h) {
                            pad_last_ktile<in0_data_format, in0_last_ktile_w>(base + h * in0_block_w * in0_stride);
                        }
                    }
                }

                cb_in0.push_back(in0_block_num_tiles);
                cb_in1.push_back(in1_block_num_tiles);
            }
        }
    }
}
