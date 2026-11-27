// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC, NoC1): row-0 segment scatter of one constant element.
//
// Raw dataflow API, no kernel_lib helper: no helper writes one constant L1 run
// to a set of (page, in-page offset) destinations of an interleaved tensor.
// zero_tile / prepare_reduce_scaler fill local L1 with a fixed pattern only;
// a `scatter_constant_run(accessor, src_l1, first_page, num_pages, offsets,
// run_bytes)` helper would close the gap.
//
// Only row 0 of each tile holds logical positions of the rank-1 tensor; it
// spans face 0 (positions 0..15, page offset 0) and face 1 (positions 16..31,
// page offset face_bytes). Every other byte of the tile is padding and is not
// written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

template <typename Accessor>
FORCE_INLINE void scatter_row0_block(
    const Accessor& dst,
    uint32_t fill_row_addr,
    uint32_t first_tile,
    uint32_t num_tiles,
    uint32_t segments_per_tile,
    uint32_t segment_bytes,
    uint32_t face_bytes) {
    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        for (uint32_t segment = 0; segment < segments_per_tile; ++segment) {
            noc_async_write_one_packet(fill_row_addr, dst.get_noc_addr(tile, segment * face_bytes), segment_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_elems = get_compile_time_arg_val(1);
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t segments_per_tile = get_compile_time_arg_val(3);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t page_size = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(1);

    // cb_fill_row is this thread's private source run: no reserve/push/wait/pop.
    const uint32_t fill_row_addr = get_write_ptr(cb_fill_row);

    {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        // Must land before the first write below reads it, or the NoC sends the
        // previous call's value left at the same CB address.
        volatile tt_l1_ptr uint32_t* fill_row = dataflow_kernel_lib::addr_to_l1_ptr(fill_row_addr);
        for (uint32_t i = 0; i < segment_elems; ++i) {
            fill_row[i] = fill_bits;
        }
    }

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        const auto dst = TensorAccessor(dst_args, dst_addr, page_size);
        const uint32_t end_tile = start_tile + core_w_tiles;
        for (uint32_t block_first = start_tile; block_first < end_tile; block_first += block_w_tiles) {
            const uint32_t block_tiles =
                (end_tile - block_first) < block_w_tiles ? (end_tile - block_first) : block_w_tiles;
            scatter_row0_block(
                dst, fill_row_addr, block_first, block_tiles, segments_per_tile, segment_bytes, face_bytes);
        }
    }

    {
        // Every write is acked before the kernel returns: the next program starts right after.
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
