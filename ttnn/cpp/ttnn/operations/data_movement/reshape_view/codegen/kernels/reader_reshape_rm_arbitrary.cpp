// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/kernels/riscv_copy.h"

using namespace tt::data_movement::common;
using namespace codegen_kernels;

// Input pages carry alignment padding, so a slab of the packed output stream is gathered from aligned windows of
// every input page it crosses, then its logical bytes are copied into one output CB slot.
struct SourceWindow {
    uint32_t page;
    uint32_t read_column;
    uint32_t read_bytes;
    uint32_t skip;
    uint32_t take;
};

template <uint32_t old_stick_bytes, uint32_t input_alignment>
FORCE_INLINE SourceWindow next_window(uint32_t flat, uint32_t left) {
    const uint32_t page = flat / old_stick_bytes;
    const uint32_t column = flat - page * old_stick_bytes;
    const uint32_t page_left = old_stick_bytes - column;
    const uint32_t take = left < page_left ? left : page_left;
    const uint32_t read_column = column & ~(input_alignment - 1);
    const uint32_t read_end = (column + take + input_alignment - 1) & ~(input_alignment - 1);
    return {page, read_column, read_end - read_column, column - read_column, take};
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t old_stick_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t old_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t new_stick_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t slab_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t slabs_per_stick = get_compile_time_arg_val(7);
    constexpr uint32_t input_alignment = get_compile_time_arg_val(8);
    constexpr uint32_t noc_max_burst_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t batch = get_compile_time_arg_val(10);
    constexpr uint32_t region_stride = get_compile_time_arg_val(11);
    constexpr auto src_args = TensorAccessorArgs<12>();

    const auto source = TensorAccessor(src_args, src_addr, old_page_bytes);
    Noc noc;
    CircularBuffer scratch_buffer(cb_scratch);
    CircularBuffer out_buffer(cb_out);

    // The scratch CB has no consumer; this kernel keeps its single page for its whole run.
    scratch_buffer.reserve_back(1);
    const uint32_t scratch_base = (scratch_buffer.get_write_ptr() + input_alignment - 1) & ~(input_alignment - 1);
    scratch_buffer.push_back(1);

    uint32_t unit = start_unit;
    uint32_t units_left = num_units;
    while (units_left > 0) {
        const uint32_t n = units_left < batch ? units_left : batch;
        out_buffer.reserve_back(n);
        const uint32_t out_base = out_buffer.get_write_ptr();

        // Every window of the batch is read before a single barrier; each slab has its own scratch region.
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t stick = (unit + j) / slabs_per_stick;
            const uint32_t column = (unit + j - stick * slabs_per_stick) * slab_bytes;
            const uint32_t stick_left = new_stick_bytes - column;
            uint32_t left = stick_left < slab_bytes ? stick_left : slab_bytes;
            uint32_t flat = stick * new_stick_bytes + column;
            const CoreLocalMem<uint8_t> region(scratch_base + j * region_stride);
            uint32_t region_offset = 0;
            while (left > 0) {
                const SourceWindow w = next_window<old_stick_bytes, input_alignment>(flat, left);
                noc.async_read(
                    source,
                    region,
                    w.read_bytes,
                    {.page_id = w.page, .offset_bytes = w.read_column},
                    {.offset_bytes = region_offset});
                region_offset += w.read_bytes;
                flat += w.take;
                left -= w.take;
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();

        // Pieces whose 16 B phases match are NOC self-reads, the rest RISC copies; both complete before the push.
        uint32_t last_store = 0;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t stick = (unit + j) / slabs_per_stick;
            const uint32_t column = (unit + j - stick * slabs_per_stick) * slab_bytes;
            const uint32_t stick_left = new_stick_bytes - column;
            uint32_t left = stick_left < slab_bytes ? stick_left : slab_bytes;
            uint32_t flat = stick * new_stick_bytes + column;
            const uint32_t region = scratch_base + j * region_stride;
            const uint32_t slot = out_base + j * slot_bytes;
            uint32_t region_offset = 0;
            uint32_t slot_offset = 0;
            while (left > 0) {
                const SourceWindow w = next_window<old_stick_bytes, input_alignment>(flat, left);
                const uint32_t dst = slot + slot_offset;
                const uint32_t src = region + region_offset + w.skip;
                if (((dst ^ src) & 15) == 0) {
                    tt_memmove<true, true, true, noc_max_burst_bytes>(noc, dst, src, w.take);
                } else {
                    last_store = riscv_copy_elements(dst, src, w.take);
                }
                region_offset += w.read_bytes;
                slot_offset += w.take;
                flat += w.take;
                left -= w.take;
            }
        }

        noc.async_read_barrier();
        if (last_store != 0) {
            drain_stores(last_store);
        }
        out_buffer.push_back(n);
        unit += n;
        units_left -= n;
    }
}
