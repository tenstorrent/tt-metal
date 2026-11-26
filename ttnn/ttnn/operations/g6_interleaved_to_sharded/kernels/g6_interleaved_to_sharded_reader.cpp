// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-fetch reader: every DRAM-interleaved page of this core's shard rectangle is read straight into
// its slot of the L1 output shard. The read response is the store, so there is no CB, no writer and no
// compute kernel.
//
// Raw dataflow API, no kernel_lib helper: the local-copy helpers program one fixed source for a loop of
// L1->L1 copies, while every page here comes from a different DRAM bank. A helper "read a page rectangle
// from an interleaved accessor into a local L1 region with a per-page segment list" would close the gap.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(1);
// Slot stride inside the shard: the full shard width, also on a clipped last shard.
constexpr uint32_t shard_cols_tiles = get_compile_time_arg_val(2);
constexpr uint32_t tensor_rows_per_image_tiles = get_compile_time_arg_val(3);
constexpr bool last_row_is_partial = get_compile_time_arg_val(4) != 0;
// Segment table of a partial tile row: at most two (offset, bytes) pairs inside the page.
constexpr uint32_t num_partial_segments = get_compile_time_arg_val(5);
constexpr uint32_t partial_seg0_offset = get_compile_time_arg_val(6);
constexpr uint32_t partial_seg0_bytes = get_compile_time_arg_val(7);
constexpr uint32_t partial_seg1_offset = get_compile_time_arg_val(8);
constexpr uint32_t partial_seg1_bytes = get_compile_time_arg_val(9);
constexpr auto input_args = TensorAccessorArgs<10>();

static_assert(num_partial_segments >= 1 && num_partial_segments <= 2);
static_assert(partial_seg0_offset + partial_seg0_bytes <= page_bytes);
static_assert(num_partial_segments < 2 || partial_seg1_offset + partial_seg1_bytes <= page_bytes);

// Every segment is at most one page, so the one-packet issue path always applies.
template <typename Accessor>
FORCE_INLINE void read_segment(const Accessor& input, uint32_t page_id, uint32_t offset, uint32_t bytes, uint32_t dst) {
    noc_async_read<page_bytes>(input.get_noc_addr(page_id, offset), dst + offset, bytes);
}

template <typename Accessor>
FORCE_INLINE void read_partial_row(
    const Accessor& input, uint32_t first_page_id, uint32_t first_dst, uint32_t block_cols) {
    for (uint32_t c = 0; c < block_cols; ++c) {
        const uint32_t page_id = first_page_id + c;
        const uint32_t dst = first_dst + c * page_bytes;
        read_segment(input, page_id, partial_seg0_offset, partial_seg0_bytes, dst);
        if constexpr (num_partial_segments == 2) {
            read_segment(input, page_id, partial_seg1_offset, partial_seg1_bytes, dst);
        }
    }
}

template <typename Accessor>
FORCE_INLINE void read_full_row(
    const Accessor& input, uint32_t first_page_id, uint32_t first_dst, uint32_t block_cols) {
    for (uint32_t c = 0; c < block_cols; ++c) {
        read_segment(input, first_page_id + c, 0, page_bytes, first_dst + c * page_bytes);
    }
}

// The block is this core's clipped shard rectangle: rows [r0, r0 + block_rows), cols [c0, c0 + block_cols).
template <typename Accessor>
FORCE_INLINE void fetch_block_into_shard(
    const Accessor& input, uint32_t shard_base, uint32_t r0, uint32_t c0, uint32_t block_rows, uint32_t block_cols) {
    for (uint32_t i = 0; i < block_rows; ++i) {
        const uint32_t r = r0 + i;
        const uint32_t first_page_id = r * tensor_cols_tiles + c0;
        const uint32_t first_dst = shard_base + i * shard_cols_tiles * page_bytes;
        if constexpr (!last_row_is_partial) {
            read_full_row(input, first_page_id, first_dst, block_cols);
        } else if constexpr (tensor_rows_per_image_tiles == 1) {
            read_partial_row(input, first_page_id, first_dst, block_cols);
        } else {
            if (r % tensor_rows_per_image_tiles == tensor_rows_per_image_tiles - 1) {
                read_partial_row(input, first_page_id, first_dst, block_cols);
            } else {
                read_full_row(input, first_page_id, first_dst, block_cols);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    // The same L1 address on every core of an L1-sharded buffer.
    const uint32_t shard_base = get_common_arg_val<uint32_t>(1);

    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t block_rows = get_arg_val<uint32_t>(2);
    const uint32_t block_cols = get_arg_val<uint32_t>(3);

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);

    {
        DeviceZoneScopedN("I2S-FETCH");
        fetch_block_into_shard(input, shard_base, r0, c0, block_rows, block_cols);
    }
    {
        DeviceZoneScopedN("I2S-BARRIER");
        noc_async_read_barrier();
    }
}
