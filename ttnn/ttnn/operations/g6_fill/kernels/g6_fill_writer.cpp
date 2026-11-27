// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC, NoC1): the only kernel of the program.
//
// Each core writes the row-0 face segments of its contiguous tile range [start_tile,
// start_tile + core_w_tiles). A rank-1 tensor [N] lives in row 0 of each tile, split across
// face 0 (positions 0..15, page offset 0) and face 1 (positions 16..31, page offset
// segment_stride_bytes); every other byte of the tile is padding and is left unwritten.
//
// Raw NoC instead of a kernel_lib helper: no dataflow helper writes one constant L1 run to a
// set of (page, in-page offset) destinations of an interleaved tensor. zero_tile writes zeros
// into local L1 only; prepare_reduce_scaler carries a 16-bit reduce pattern, not 32 bits.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

template <uint32_t segment_elems>
FORCE_INLINE void build_fill_row(uint32_t fill_row_addr, uint32_t fill_bits) {
    volatile tt_l1_ptr uint32_t* fill_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_row_addr);
    for (uint32_t i = 0; i < segment_elems; ++i) {
        fill_row[i] = fill_bits;
    }
}

template <uint32_t segment_bytes, uint32_t segments_per_tile, uint32_t segment_stride_bytes, typename Accessor>
FORCE_INLINE void scatter_row0_block(
    const Accessor& dst, uint32_t fill_row_addr, uint32_t first_tile, uint32_t num_tiles) {
    const uint32_t end_tile = first_tile + num_tiles;
    for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        for (uint32_t segment = 0; segment < segments_per_tile; ++segment) {
            noc_async_write_one_packet(
                fill_row_addr, dst.get_noc_addr(tile, segment * segment_stride_bytes), segment_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t segments_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t segment_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    // Common runtime-arg slots; the host patches them on its cached descriptor every call.
    constexpr uint32_t common_arg_dst_addr = 0;
    constexpr uint32_t common_arg_fill_bits = 1;

    constexpr uint32_t segment_elems = segment_bytes / sizeof(uint32_t);
    static_assert(segment_bytes % sizeof(uint32_t) == 0);
    // Blackhole DRAM writes need L1 source offset == destination offset (mod 16).
    static_assert(segment_stride_bytes % 16 == 0);
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE);
    static_assert(block_w_tiles >= 1);

    const uint32_t fill_row_addr = get_write_ptr(cb_fill_row);

    {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        // The row must be complete before the first write reads it: the CB keeps the previous
        // call's value at the same address.
        build_fill_row<segment_elems>(fill_row_addr, get_common_arg_val<uint32_t>(common_arg_fill_bits));
    }

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        const uint32_t dst_addr = get_common_arg_val<uint32_t>(common_arg_dst_addr);
        const uint32_t start_tile = get_arg_val<uint32_t>(0);
        const uint32_t core_w_tiles = get_arg_val<uint32_t>(1);
        const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

        const uint32_t end_tile = start_tile + core_w_tiles;
        for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_w_tiles) {
            const uint32_t remaining = end_tile - block_start;
            const uint32_t block_tiles = remaining < block_w_tiles ? remaining : block_w_tiles;
            scatter_row0_block<segment_bytes, segments_per_tile, segment_stride_bytes>(
                dst, fill_row_addr, block_start, block_tiles);
        }
    }

    {
        DeviceZoneScopedN("G6FILL-DRAIN");
        // The next program starts as soon as this kernel returns; every write must be acked.
        noc_async_write_barrier();
    }
}
