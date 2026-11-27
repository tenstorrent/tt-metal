// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer: materializes one segment of the fill element in L1 and writes it, as the NoC source,
// to every logical segment of this core's range of the output. No tensor is read.
//
// store_block / drain_writes use raw noc_async_write + TensorAccessor::get_noc_addr: no kernel_lib helper
// writes one L1 source to a list of (page, intra-page offset) destinations of an interleaved tensor, and
// noc_inline_dw_write does not reach DRAM. The segment loop is the block operation.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    // Longest contiguous logical run in an output page: one face row for TILE.
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t segments_per_page = get_compile_time_arg_val(2);
    // Byte distance from one segment of a page to the next (face 0 row 0 -> face 1 row 0 for TILE).
    constexpr uint32_t segment_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    // Common RT args: never part of the program hash, so a cached program writes the current value.
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t e_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t block_segments = get_common_arg_val<uint32_t>(2);

    const uint32_t seg_start = get_arg_val<uint32_t>(0);
    const uint32_t seg_count = get_arg_val<uint32_t>(1);

    // Scratch NoC source: this kernel is the CB's only user, so no reserve/push handshake.
    const uint32_t fill_row_addr = get_write_ptr(cb_fill_row);
    {
        DeviceZoneScopedN("G6FILL-MATERIALIZE");
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_row_addr, segment_bytes, e_bits);
    }

    const auto out = TensorAccessor(out_args, out_addr, out_page_bytes);
    const uint32_t seg_end = seg_start + seg_count;
    {
        DeviceZoneScopedN("G6FILL-STORE");
        for (uint32_t block_start = seg_start; block_start < seg_end; block_start += block_segments) {
            const uint32_t block_end = block_start + block_segments < seg_end ? block_start + block_segments : seg_end;
            // No barrier inside or between blocks: the source is never rewritten.
            for (uint32_t s = block_start; s < block_end; ++s) {
                const uint32_t page = s / segments_per_page;
                const uint32_t offset = (s % segments_per_page) * segment_stride_bytes;
                noc_async_write(fill_row_addr, out.get_noc_addr(page, offset), segment_bytes);
            }
        }
    }
    {
        // Every write is acked before return: the next program starts right after this one.
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
