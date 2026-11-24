// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gathered-prefix untilize writer (untilize with gathered_prefix_tensor). Reads the device-held chunk start itself
// and walks the same tile rows as reader_untilize_extent.cpp (valid tile row v -> core v % num_cores), writing each
// tile row's 32 RM sticks at their place in the full-size output (rows outside the gathered prefix are not written).
// One write barrier per tile row, as writer_untilize_interleaved.cpp.
//
// CT args: cb_out, cb_scratch, Wt, row_size_bytes, seg_tile_rows, num_segments, slab_rows, total_rows,
//          TensorAccessorArgs(output), TensorAccessorArgs(extent)
// RT args: output address, extent address, core index, number of cores
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t wt = get_compile_time_arg_val(2);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t seg_tile_rows = get_compile_time_arg_val(4);
    constexpr uint32_t num_segments = get_compile_time_arg_val(5);
    constexpr uint32_t slab_rows = get_compile_time_arg_val(6);
    constexpr uint32_t total_rows = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();
    constexpr auto ext_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_height = 32;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t ext_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_idx = get_arg_val<uint32_t>(2);
    const uint32_t num_cores = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer scratch_cb(cb_scratch);

    // the writer's own L1 scratch for the extent read (never pushed)
    scratch_cb.reserve_back(1);
    const uint32_t start =
        trace_metadata::read_metadata_scalar_u32(noc, ext_args, ext_addr, scratch_cb.get_write_ptr());
    uint32_t g = ((start + slab_rows) / slab_rows) * slab_rows;
    if ((start + slab_rows) % slab_rows) {
        g += slab_rows;
    }
    if (g > total_rows) {
        g = total_rows;
    }
    const uint32_t valid_per_seg = g / num_segments / tile_height;
    const uint32_t valid_total = valid_per_seg * num_segments;
    const uint32_t count = core_idx < valid_total ? (valid_total - core_idx + num_cores - 1) / num_cores : 0;

    const uint32_t destination_page_size = dst_args.get_aligned_page_size();
    const auto d = TensorAccessor(dst_args, dst_addr, destination_page_size);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t v = core_idx + i * num_cores;
        const uint32_t tile_row = (v / valid_per_seg) * seg_tile_rows + v % valid_per_seg;
        out_cb.wait_front(wt);
        uint32_t l1_read_offset = 0;
        uint32_t stick = tile_row * tile_height;
        for (uint32_t row = 0; row < tile_height; ++row) {
            noc.async_write(
                out_cb, d, row_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = stick, .offset_bytes = 0});
            l1_read_offset += row_size_bytes;
            stick++;
        }
        noc.async_write_barrier();
        out_cb.pop_front(wt);
    }
}
