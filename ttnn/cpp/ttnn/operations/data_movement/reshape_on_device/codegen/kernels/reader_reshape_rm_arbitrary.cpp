// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One-program arbitrary-width ROW_MAJOR reshape reader (BATCHED).
//
// RM pages contain alignment padding, while reshape operates on the packed
// logical byte stream.  Each work unit assembles one fixed-size output slab by
// reading aligned windows from as many source pages as it crosses, then copying
// only logical bytes into the producer CB.
//
// Batching (the perf lever): the original issued one noc_async_read_barrier per
// source window — for a wide->narrow repage (tiny output stick from one source
// stick) that is 1 read + 1 barrier per unit across tens of thousands of units.
// This version processes NABATCH units per iteration: it reads ALL their source
// windows into NABATCH distinct scratch regions with NO per-window barrier, then
// ONE barrier, then memmoves/pads all NABATCH into NABATCH cb_out slots and
// pushes them together. Barrier count drops ~NABATCH x. Byte-identical: the
// per-window read/memmove geometry is unchanged (recomputed identically in the
// read and memmove phases); disjoint async reads before one barrier == per-read
// barriers. Reduces to the original at NABATCH=1.
//
// CT: cb_out, cb_scratch, old_stick_bytes, old_page_bytes, new_stick_bytes,
//     slab_bytes, slab_slot_bytes, slabs_per_output, input_alignment,
//     noc_max_burst_bytes, new_page_bytes, NABATCH, region_stride,
//     TensorAccessorArgs(input)
// RT: src_addr, start_unit, num_units
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t old_stick_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t old_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t new_stick_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t slab_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t slab_slot_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t slabs_per_output = get_compile_time_arg_val(7);
    constexpr uint32_t input_alignment = get_compile_time_arg_val(8);
    constexpr uint32_t noc_max_burst_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t new_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t NABATCH = get_compile_time_arg_val(11);
    constexpr uint32_t region_stride = get_compile_time_arg_val(12);
    constexpr auto src_args = TensorAccessorArgs<13>();

    const auto source = TensorAccessor(src_args, src_addr, old_page_bytes);
    Noc noc;  // for the window reads and the tt_memmove bulk L1->L1 copy below
    CircularBuffer scratch_buffer(cb_scratch);
    CircularBuffer out_buffer(cb_out);

    // Claim NABATCH scratch regions for this kernel's whole lifetime (no
    // downstream consumer). scratch_base is aligned; region j is at
    // scratch_base + j*region_stride (region_stride is an input_alignment
    // multiple, so every region and every intra-region aligned read stays
    // input_alignment-aligned). Regions are reused every batch iteration.
    scratch_buffer.reserve_back(NABATCH);
    const uint32_t scratch_base_raw = scratch_buffer.get_write_ptr();
    scratch_buffer.push_back(NABATCH);
    const uint32_t scratch_base = (scratch_base_raw + input_alignment - 1) & ~(input_alignment - 1);

    uint32_t units_left = num_units;
    uint32_t unit = start_unit;
    while (units_left > 0) {
        const uint32_t B = (units_left < NABATCH) ? units_left : NABATCH;
        out_buffer.reserve_back(B);
        const uint32_t out_base = out_buffer.get_write_ptr();

        // Phase 1: issue every window read for all B units (NO barrier).
        for (uint32_t j = 0; j < B; ++j) {
            const uint32_t u = unit + j;
            const uint32_t output_page = u / slabs_per_output;
            const uint32_t slab_index = u - output_page * slabs_per_output;
            const uint32_t output_column = slab_index * slab_bytes;
            const uint32_t logical_bytes =
                ((new_stick_bytes - output_column) < slab_bytes) ? (new_stick_bytes - output_column) : slab_bytes;
            const CoreLocalMem<uint8_t> region(scratch_base + j * region_stride);

            uint32_t flat = output_page * new_stick_bytes + output_column;
            uint32_t woff = 0;
            uint32_t left = logical_bytes;
            while (left > 0) {
                const uint32_t source_page = flat / old_stick_bytes;
                const uint32_t source_column = flat - source_page * old_stick_bytes;
                const uint32_t page_left = old_stick_bytes - source_column;
                const uint32_t take = left < page_left ? left : page_left;
                const uint32_t read_column = source_column & ~(input_alignment - 1);
                const uint32_t read_end = (source_column + take + input_alignment - 1) & ~(input_alignment - 1);
                const uint32_t read_bytes = read_end - read_column;
                if (read_bytes > noc_max_burst_bytes) {
                    return;  // host geometry guarantees this cannot happen
                }
                noc.async_read(
                    source,
                    region,
                    read_bytes,
                    {.page_id = source_page, .offset_bytes = read_column},
                    {.offset_bytes = woff});
                woff += read_bytes;  // aligned -> next dest stays aligned
                flat += take;
                left -= take;
            }
        }
        noc.async_read_barrier();  // ONE barrier for the whole batch

        // Phase 2: memmove logical bytes + zero pad for each unit. Recomputes the
        // identical window geometry so woff matches phase 1 exactly.
        for (uint32_t j = 0; j < B; ++j) {
            const uint32_t u = unit + j;
            const uint32_t output_page = u / slabs_per_output;
            const uint32_t slab_index = u - output_page * slabs_per_output;
            const uint32_t output_column = slab_index * slab_bytes;
            const uint32_t logical_bytes =
                ((new_stick_bytes - output_column) < slab_bytes) ? (new_stick_bytes - output_column) : slab_bytes;
            const bool final = output_column + logical_bytes == new_stick_bytes;
            const uint32_t page_span = final ? (new_page_bytes - output_column) : slab_bytes;
            const uint32_t region = scratch_base + j * region_stride;
            const uint32_t output = out_base + j * slab_slot_bytes;

            uint32_t flat = output_page * new_stick_bytes + output_column;
            uint32_t woff = 0;
            uint32_t slab_column = 0;
            uint32_t left = logical_bytes;
            while (left > 0) {
                const uint32_t source_page = flat / old_stick_bytes;
                const uint32_t source_column = flat - source_page * old_stick_bytes;
                const uint32_t page_left = old_stick_bytes - source_column;
                const uint32_t take = left < page_left ? left : page_left;
                const uint32_t read_column = source_column & ~(input_alignment - 1);
                const uint32_t read_end = (source_column + take + input_alignment - 1) & ~(input_alignment - 1);
                const uint32_t read_bytes = read_end - read_column;
                tt_memmove<false, false, true, noc_max_burst_bytes>(
                    noc, output + slab_column, region + woff + (source_column - read_column), take);
                woff += read_bytes;
                slab_column += take;
                flat += take;
                left -= take;
            }
            CoreLocalMem<volatile uint8_t> pad(output);
            for (uint32_t b = logical_bytes; b < page_span; ++b) {
                pad[b] = 0;
            }
        }

        out_buffer.push_back(B);
        unit += B;
        units_left -= B;
    }
}
