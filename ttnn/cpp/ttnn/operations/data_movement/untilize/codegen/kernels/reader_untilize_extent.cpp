// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gathered-prefix untilize reader (untilize with gathered_prefix_tensor). Reads the device-held chunk start, derives
// the gathered extent g = min(round_up(start + slab, slab), total_rows) and the valid tile rows of each segment
// (g / num_segments / 32), tells the compute kernel how many tile rows this core untilizes (one-page mailbox CB), then
// reads them: valid tile row v (segment-major: v / valid_per_seg = segment, v % valid_per_seg = row in it) goes to
// core v % num_cores, so a short prefix still spreads over every core.
//
// CT args: cb_in, cb_meta, Wt, seg_tile_rows, num_segments, slab_rows, total_rows, TensorAccessorArgs(input),
//          TensorAccessorArgs(extent)
// RT args: input address, extent address, core index, number of cores
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t wt = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t num_segments = get_compile_time_arg_val(4);
    constexpr uint32_t slab_rows = get_compile_time_arg_val(5);
    constexpr uint32_t total_rows = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto ext_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t ext_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_idx = get_arg_val<uint32_t>(2);
    const uint32_t num_cores = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer in_cb(cb_in);
    CircularBuffer meta_cb(cb_meta);

    meta_cb.reserve_back(1);
    const uint32_t scratch = meta_cb.get_write_ptr();
    const uint32_t start = trace_metadata::read_metadata_scalar_u32(noc, ext_args, ext_addr, scratch);
    // the extent high_bw_all_gather gathers: whole slabs covering [0, start + slab), at most the whole height
    uint32_t g = ((start + slab_rows) / slab_rows) * slab_rows;
    if ((start + slab_rows) % slab_rows) {
        g += slab_rows;
    }
    if (g > total_rows) {
        g = total_rows;
    }
    const uint32_t valid_per_seg = g / num_segments / 32;  // tile rows of each segment to untilize
    const uint32_t valid_total = valid_per_seg * num_segments;
    const uint32_t count = core_idx < valid_total ? (valid_total - core_idx + num_cores - 1) / num_cores : 0;
    volatile uint32_t* mailbox = reinterpret_cast<volatile uint32_t*>(scratch);
    mailbox[0] = count;
    mailbox[1] = valid_per_seg;
    clobber_all_memory();
    meta_cb.push_back(1);

    const uint32_t page_size = src_args.get_aligned_page_size();
    const uint32_t cb_page_size = get_local_cb_interface(cb_in).fifo_page_size << cb_addr_shift;
    const uint32_t read_size = page_size < cb_page_size ? page_size : cb_page_size;
    const auto s = TensorAccessor(src_args, src_addr, page_size);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t v = core_idx + i * num_cores;
        const uint32_t tile_row = (v / valid_per_seg) * seg_tile_rows + v % valid_per_seg;
        in_cb.reserve_back(wt);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < wt; ++t) {
            noc.async_read(
                s, in_cb, read_size, {.page_id = tile_row * wt + t, .offset_bytes = 0}, {.offset_bytes = l1_offset});
            l1_offset += cb_page_size;
        }
        noc.async_read_barrier();
        in_cb.push_back(wt);
    }
}
