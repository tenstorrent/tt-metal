// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Function reads from RM and writes to RM

Compile-time arguments
0. src_aligned_to_64
1. src_aligned_to_16
2. cb_id_in0
3. cb_id_in1
4. source_page_size_bytes
5. dest_page_size_bytes
6. num_dest_write_slots
7. dest_slot_size_bytes
8. dest_write_size_bytes
9+. TensorAccessorArgs for src, then dst

Runtime arguments
0. src_addr
1. dst_addr
2. source_read_size_bytes
3. read_start_page
4. read_end_page
5. write_start_page
6. write_start_offset
7. nop
*/
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

FORCE_INLINE void acquire_dest_slot(Noc noc, uint32_t& slots_in_flight, const uint32_t num_dest_write_slots) {
    if (slots_in_flight >= num_dest_write_slots) {
        noc.async_write_barrier();
        slots_in_flight = 0;
    }
}

FORCE_INLINE void advance_dest_slot(
    uint32_t& dest_slot, uint32_t& slots_in_flight, const uint32_t num_dest_write_slots) {
    slots_in_flight++;
    dest_slot = (dest_slot + 1) % num_dest_write_slots;
}

void kernel_main() {
    // We are guaranteed to be in 2D going to 2D

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    // If DDR this is source_page_size_bytes + 64 (rounded up to next 64B), if L1 this is source_page_size_bytes + 16
    // (rounded up to next 16B)
    const uint32_t source_read_size_bytes = get_arg_val<uint32_t>(2);
    const uint32_t read_start_page = get_arg_val<uint32_t>(3);
    const uint32_t read_end_page = get_arg_val<uint32_t>(4);
    const uint32_t write_start_page = get_arg_val<uint32_t>(5);
    const uint32_t write_start_offset = get_arg_val<uint32_t>(6);
    const uint32_t nop = get_arg_val<uint32_t>(7);

    constexpr bool src_aligned_to_64 = get_compile_time_arg_val(0) == 1;
    constexpr bool src_aligned_to_16 = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t source_page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t dest_page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_dest_write_slots = get_compile_time_arg_val(6);
    constexpr uint32_t dest_slot_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t dest_write_size_bytes = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    // Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this
    // core
    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(src_args, src_addr);
    const auto d = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer cb_in0(cb_id_in0);
    DataflowBuffer cb_in1(cb_id_in1);
    uint32_t read_offset = 0;
    uint32_t write_page = write_start_page;
    uint32_t readable = 0;
    uint32_t end_to_write = 0;
    uint32_t writable = dest_page_size_bytes - write_start_offset;
    // Ring base from DataflowBuffer; slot offsets applied manually below.
    cb_in0.reserve_back(1);
    cb_in1.reserve_back(1);
    const uint32_t source_buffer = cb_in0.get_write_ptr();
    const uint32_t dest_buffer = cb_in1.get_write_ptr();
    cb_in1.push_back(1);
    cb_in0.push_back(1);

    uint64_t dst_noc_addr = d.get_noc_addr(write_page);
    uint64_t write_offset = (dst_noc_addr & OFFSET_16) + write_start_offset;
    uint64_t begin_write_offset = write_offset;
    constexpr bool can_be_clean = ((source_page_size_bytes % 16) == 0 && (dest_page_size_bytes % 16) == 0);
    uint64_t dst_noc_addr_offset = 0;

    uint32_t dest_slot = 0;
    uint32_t slots_in_flight = 0;

    for (uint32_t i = read_start_page; i < read_end_page; i++) {
        // Drain any prior iteration's writes that read source_buffer before this iteration's read
        // overwrites it: source_buffer is a single fixed CB slot reused every iteration, and the
        // clean-path enhanced_noc_async_write below reads it asynchronously.  Without this flush the
        // next read can overwrite the slot while the previous write's source-read is still in flight
        // (write-after-read), corrupting the output.  Uses the object-noc flush to match the writes.
        noc.async_writes_flushed();
        // Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i, 0);

        if constexpr (src_aligned_to_64 || ((!src_args.is_dram) && src_aligned_to_16)) {  // Aligned to 64 bytes or 16
                                                                                          // bytes but L1
            tt::data_movement::common::enhanced_noc_async_read<source_page_size_bytes, false>(
                noc, src_noc_addr, source_buffer, source_page_size_bytes);
            read_offset = 0;
        } else if constexpr (src_args.is_dram) {  // DDR but not aligned to 64 (potentially also not aligned to 16)
            tt::data_movement::common::enhanced_noc_async_read<(source_page_size_bytes + 128), false>(
                noc, src_noc_addr & MASK_64, source_buffer, source_read_size_bytes);
            read_offset = src_noc_addr & OFFSET_64;
        } else {  // L1 but not aligned to 16
            tt::data_movement::common::enhanced_noc_async_read<(source_page_size_bytes + 128), false>(
                noc, src_noc_addr & MASK_16, source_buffer, source_read_size_bytes);
            read_offset = src_noc_addr & OFFSET_16;
        }

        readable = source_page_size_bytes;
        noc.async_read_barrier();

        // Write to dest
        while (readable > 0) {
            if constexpr (can_be_clean) {
                noc.async_write_barrier();
            }
            if (readable < writable) {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, readable);
                    dst_noc_addr_offset = dst_noc_addr_offset + readable;
                } else {
                    acquire_dest_slot(noc, slots_in_flight, num_dest_write_slots);
                    const uint32_t slot_base = dest_buffer + dest_slot * dest_slot_size_bytes;
                    // use_read_datamover=true: sync via read barrier so prior DRAM dest writes stay in flight.
                    tt::data_movement::common::tt_memmove<false, false, true, dest_page_size_bytes>(
                        noc, slot_base + write_offset, source_buffer + read_offset, readable);
                    if (i == read_end_page - 1) {
                        const uint32_t bytes_to_flush = end_to_write + readable;
                        tt::data_movement::common::enhanced_noc_async_write<dest_write_size_bytes, false>(
                            noc, slot_base + begin_write_offset, dst_noc_addr, bytes_to_flush);
                        noc.async_write_barrier();
                        return;
                    }
                    write_offset = write_offset + readable;
                    end_to_write = end_to_write + readable;
                }
                writable = writable - readable;
                readable = 0;

            } else if (readable == writable) {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, readable);
                } else {
                    acquire_dest_slot(noc, slots_in_flight, num_dest_write_slots);
                    const uint32_t slot_base = dest_buffer + dest_slot * dest_slot_size_bytes;
                    // use_read_datamover=true: sync via read barrier so prior DRAM dest writes stay in flight.
                    tt::data_movement::common::tt_memmove<false, false, true, dest_page_size_bytes>(
                        noc, slot_base + write_offset, source_buffer + read_offset, readable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_write_size_bytes, false>(
                        noc, slot_base + begin_write_offset, dst_noc_addr, dest_write_size_bytes);
                    advance_dest_slot(dest_slot, slots_in_flight, num_dest_write_slots);
                }
                dst_noc_addr_offset = 0;

                writable = dest_page_size_bytes;
                readable = 0;
                if (i == read_end_page - 1) {
                    noc.async_write_barrier();
                    return;
                }
                write_page++;
                dst_noc_addr = d.get_noc_addr(write_page);
                if constexpr (!can_be_clean) {
                    end_to_write = 0;
                    write_offset = dst_noc_addr & OFFSET_16;
                    begin_write_offset = write_offset;
                }
            } else {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, writable);
                } else {
                    acquire_dest_slot(noc, slots_in_flight, num_dest_write_slots);
                    const uint32_t slot_base = dest_buffer + dest_slot * dest_slot_size_bytes;
                    // use_read_datamover=true: sync via read barrier so prior DRAM dest writes stay in flight.
                    tt::data_movement::common::tt_memmove<false, false, true, dest_page_size_bytes>(
                        noc, slot_base + write_offset, source_buffer + read_offset, writable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_write_size_bytes, false>(
                        noc, slot_base + begin_write_offset, dst_noc_addr, dest_write_size_bytes);
                    advance_dest_slot(dest_slot, slots_in_flight, num_dest_write_slots);
                }
                // writable < readable
                readable = readable - writable;
                read_offset = read_offset + writable;
                write_page++;
                dst_noc_addr_offset = 0;
                dst_noc_addr = d.get_noc_addr(write_page);
                if constexpr (!can_be_clean) {
                    end_to_write = 0;
                    write_offset = dst_noc_addr & OFFSET_16;
                    begin_write_offset = write_offset;
                }
                writable = dest_page_size_bytes;
            }
        }
    }
    noc.async_write_barrier();
    return;
}
