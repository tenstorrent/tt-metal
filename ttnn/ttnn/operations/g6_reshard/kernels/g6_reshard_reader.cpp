// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_reshard PULL issuer. One source, instantiated num_issuers times per out-block core (issuer 0 on
// RISCV_1/NOC_0, issuer 1 on RISCV_0/NOC_1). Each issuer pulls one row band of this core's out block straight
// from the in0 shards that hold it: every run lands at its final slot of the out shard, so there is no CB.
//
// Raw NoC calls instead of a kernel_lib helper: local_copy_helpers' read-state helpers are self-aimed and
// single-packet with a compile-time size, while these runs are remote and runtime-length (1 tile .. a shard).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
FORCE_INLINE uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

}  // namespace

void kernel_main() {
    constexpr uint32_t issuer_idx = get_compile_time_arg_val(0);
    constexpr uint32_t num_issuers = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in_shard_row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t in_shard_col_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in_block_cols = get_compile_time_arg_val(6);
    constexpr uint32_t out_shard_row_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t out_shard_col_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tile_stride_bytes = get_compile_time_arg_val(9);
    constexpr auto in0_args = TensorAccessorArgs<10>();

    static_assert(issuer_idx < num_issuers, "issuer_idx out of range");

    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t block_row = get_arg_val<uint32_t>(0);
    const uint32_t block_col = get_arg_val<uint32_t>(1);

    // The out block, clipped to the tensor on a tail block row / column.
    const uint32_t block_row_begin = block_row * out_shard_row_tiles;
    const uint32_t block_row_end = min_u32(block_row_begin + out_shard_row_tiles, tensor_row_tiles);
    const uint32_t block_col_begin = block_col * out_shard_col_tiles;
    const uint32_t block_col_end = min_u32(block_col_begin + out_shard_col_tiles, tensor_col_tiles);

    const uint32_t band_height_tiles = (block_row_end - block_row_begin + num_issuers - 1) / num_issuers;
    const uint32_t band_row_begin = min_u32(block_row_begin + issuer_idx * band_height_tiles, block_row_end);
    const uint32_t band_row_end = min_u32(band_row_begin + band_height_tiles, block_row_end);

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_stride_bytes);

    {
        DeviceZoneScopedN("G6R-PULL-ISSUE");
        if (band_row_begin < band_row_end) {
            const uint32_t src_block_row_first = band_row_begin / in_shard_row_tiles;
            const uint32_t src_block_row_last = (band_row_end - 1) / in_shard_row_tiles;
            const uint32_t src_block_col_first = block_col_begin / in_shard_col_tiles;
            const uint32_t src_block_col_last = (block_col_end - 1) / in_shard_col_tiles;

            for (uint32_t src_block_row = src_block_row_first; src_block_row <= src_block_row_last; ++src_block_row) {
                const uint32_t src_row_origin = src_block_row * in_shard_row_tiles;
                const uint32_t h0 = max_u32(band_row_begin, src_row_origin);
                const uint32_t h1 = min_u32(band_row_end, src_row_origin + in_shard_row_tiles);

                for (uint32_t src_block_col = src_block_col_first; src_block_col <= src_block_col_last;
                     ++src_block_col) {
                    const uint32_t src_col_origin = src_block_col * in_shard_col_tiles;
                    const uint32_t w0 = max_u32(block_col_begin, src_col_origin);
                    const uint32_t w1 = min_u32(block_col_end, src_col_origin + in_shard_col_tiles);
                    const uint32_t run_width = w1 - w0;

                    const uint32_t shard_id = src_block_row * in_block_cols + src_block_col;
                    ASSERT(shard_id < in0.dspec().num_shards());
                    const uint64_t shard_noc_addr = in0.get_shard_noc_addr(shard_id, 0, noc_index);
                    noc_async_read_set_state(shard_noc_addr, noc_index);

                    // Strides are the allocated shard widths, also on a tail shard (in-shard page order).
                    uint32_t src_l1 =
                        static_cast<uint32_t>(shard_noc_addr) +
                        ((h0 - src_row_origin) * in_shard_col_tiles + (w0 - src_col_origin)) * tile_stride_bytes;
                    uint32_t dst_l1 =
                        out_addr +
                        ((h0 - block_row_begin) * out_shard_col_tiles + (w0 - block_col_begin)) * tile_stride_bytes;

                    // Full rows of both allocated shards are contiguous on both ends: one run for the whole
                    // intersection. A tail intersection is narrower than the allocated width, so it never merges.
                    if (run_width == in_shard_col_tiles && run_width == out_shard_col_tiles) {
                        noc_async_read_with_state(src_l1, dst_l1, (h1 - h0) * run_width * tile_stride_bytes, noc_index);
                    } else {
                        const uint32_t run_bytes = run_width * tile_stride_bytes;
                        for (uint32_t h = h0; h < h1; ++h) {
                            noc_async_read_with_state(src_l1, dst_l1, run_bytes, noc_index);
                            src_l1 += in_shard_col_tiles * tile_stride_bytes;
                            dst_l1 += out_shard_col_tiles * tile_stride_bytes;
                        }
                    }
                }
            }
        }
    }

    {
        DeviceZoneScopedN("G6R-PULL-DRAIN");
        noc_async_read_barrier(noc_index);
    }
}
