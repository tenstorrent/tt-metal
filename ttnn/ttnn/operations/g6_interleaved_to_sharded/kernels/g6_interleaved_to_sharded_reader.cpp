// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-fetch reader: this core reads its own shard rectangle of the DRAM-interleaved input
// straight into its L1 shard. No CB, no compute, no writer: the read response is the result.
//
// Raw noc_async_read instead of a kernel_lib helper: kernel_lib has no DRAM -> L1 page-read
// helper. The local-copy helpers are L1 -> L1 on this core only, and eltwise copy routes tiles
// through unpack/pack, which may rewrite NaN payloads and denormals.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

// G6I2S_ABLATE_READS (measurement only) drops the NoC reads and keeps the loop, the address
// generation and the barrier: the kernel's sync-only floor.
template <uint32_t max_bytes>
FORCE_INLINE void read_span(uint64_t src_noc_addr, uint32_t dst_l1_addr, uint32_t size) {
#ifndef G6I2S_ABLATE_READS
    noc_async_read<max_bytes>(src_noc_addr, dst_l1_addr, size);
#endif
}

// Issues every span read of the block back to back, with no barrier or handshake between pages.
// When trim_span_bytes != 0, a page in the last tile row of an image moves only the first
// trim_span_bytes of faces 0 and 1 (its logical rows): two spans, or with trim_single_span one
// span from the page start through the end of face 1's span. Every other page moves whole.
template <
    uint32_t tensor_cols_tiles,
    uint32_t tiles_h_per_image,
    uint32_t shard_cols_tiles,
    uint32_t page_size,
    uint32_t trim_span_bytes,
    uint32_t face1_offset_bytes,
    bool trim_single_span,
    typename Accessor>
FORCE_INLINE void fetch_shard_block(
    const Accessor& input,
    uint32_t block_dst_addr,
    uint32_t block_row_start,
    uint32_t block_rows,
    uint32_t block_col_start,
    uint32_t block_cols) {
    constexpr uint32_t shard_row_stride_bytes = shard_cols_tiles * page_size;
    uint32_t row_dst = block_dst_addr;
    for (uint32_t r = 0; r < block_rows; ++r) {
        const uint32_t tensor_row = block_row_start + r;
        uint32_t page_id = tensor_row * tensor_cols_tiles + block_col_start;
        uint32_t dst = row_dst;
        bool trimmed = false;
        if constexpr (trim_span_bytes != 0) {
            trimmed = (tensor_row + 1) % tiles_h_per_image == 0;
        }
        if (trimmed) {
            for (uint32_t c = 0; c < block_cols; ++c, ++page_id, dst += page_size) {
                if constexpr (trim_single_span) {
                    constexpr uint32_t span_bytes = face1_offset_bytes + trim_span_bytes;
                    read_span<span_bytes>(input.get_noc_addr(page_id), dst, span_bytes);
                } else {
                    read_span<trim_span_bytes>(input.get_noc_addr(page_id), dst, trim_span_bytes);
                    read_span<trim_span_bytes>(
                        input.get_noc_addr(page_id, face1_offset_bytes), dst + face1_offset_bytes, trim_span_bytes);
                }
            }
        } else {
            for (uint32_t c = 0; c < block_cols; ++c, ++page_id, dst += page_size) {
                read_span<page_size>(input.get_noc_addr(page_id), dst, page_size);
            }
        }
        // The nominal shard stride, not block_cols: a ragged last shard keeps the shard's row pitch.
        row_dst += shard_row_stride_bytes;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_h_per_image = get_compile_time_arg_val(1);
    constexpr uint32_t shard_cols_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t trim_span_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t face1_offset_bytes = get_compile_time_arg_val(5);
    constexpr bool trim_single_span = get_compile_time_arg_val(6) != 0;
    constexpr auto input_args = TensorAccessorArgs<7>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_shard_base_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t block_row_start = get_arg_val<uint32_t>(0);
    const uint32_t block_rows = get_arg_val<uint32_t>(1);
    const uint32_t block_col_start = get_arg_val<uint32_t>(2);
    const uint32_t block_cols = get_arg_val<uint32_t>(3);
    const uint32_t block_local_start_page = get_arg_val<uint32_t>(4);

    // The whole resident shard is the block: one block on an owner core, none on a core of the
    // grid past the last shard.
    const uint32_t num_blocks_this_core = (block_rows != 0 && block_cols != 0) ? 1 : 0;
    if (num_blocks_this_core == 0) {
        DeviceZoneScopedN("G6I2S-IDLE");
        return;
    }

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        {
            DeviceZoneScopedN("G6I2S-FETCH");
            fetch_shard_block<
                tensor_cols_tiles,
                tiles_h_per_image,
                shard_cols_tiles,
                page_size,
                trim_span_bytes,
                face1_offset_bytes,
                trim_single_span>(
                input,
                output_shard_base_addr + block_local_start_page * page_size,
                block_row_start,
                block_rows,
                block_col_start,
                block_cols);
        }
        {
            // The kernel's last NoC action: nothing may be in flight when the next program starts.
            DeviceZoneScopedN("G6I2S-DRAIN");
            noc_async_read_barrier();
        }
    }
}
