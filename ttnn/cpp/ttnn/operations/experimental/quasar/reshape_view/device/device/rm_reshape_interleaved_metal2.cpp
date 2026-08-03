// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Metal-2.0 (Quasar) variant of rm_reshape_interleaved.cpp. Identical RM->RM repaging logic
(handles source_page_size != dest_page_size), but wired for the Metal-2 ProgramSpec path:
  - src/dst buffers come from bound tensor parameters (tensor::src / tensor::dst), NOT from
    RTA addresses + TensorAccessorArgs;
  - the two L1 staging buffers are private node-local scratchpads (scratch::src_stage /
    scratch::dst_stage), NOT DFBs -- a DM kernel that both fills and drains a DFB is an
    unsupported producer+consumer self-loop on Gen2/Quasar (program_spec.cpp:1309);
  - args are named via get_arg(args::...).

Compile args (named): src_aligned_to_64, src_aligned_to_16, src_is_dram, source_page_size_bytes,
dest_page_size_bytes.
Runtime args (named): source_read_size_bytes, read_start_page, read_end_page, write_start_page,
write_start_offset, nop.
*/
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    const uint32_t source_read_size_bytes = get_arg(args::source_read_size_bytes);
    const uint32_t read_start_page = get_arg(args::read_start_page);
    const uint32_t read_end_page = get_arg(args::read_end_page);
    const uint32_t write_start_page = get_arg(args::write_start_page);
    const uint32_t write_start_offset = get_arg(args::write_start_offset);
    const uint32_t nop = get_arg(args::nop);

    constexpr bool src_aligned_to_64 = get_arg(args::src_aligned_to_64) == 1;
    constexpr bool src_aligned_to_16 = get_arg(args::src_aligned_to_16) == 1;
    constexpr bool src_is_dram = get_arg(args::src_is_dram) == 1;
    constexpr uint32_t source_page_size_bytes = get_arg(args::source_page_size_bytes);
    constexpr uint32_t dest_page_size_bytes = get_arg(args::dest_page_size_bytes);

    // Idle core: nothing to do (matches the legacy kernel's nop short-circuit).
    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(tensor::src);
    const auto d = TensorAccessor(tensor::dst);

    Noc noc;
    // Private node-local L1 staging (raw memory, no producer/consumer credit semantics). These
    // replace the legacy cb_in0/cb_in1 single-kernel scratch CBs (which would be Gen2 self-loops).
    Scratchpad<uint8_t> src_stage(scratch::src_stage);
    Scratchpad<uint8_t> dst_stage(scratch::dst_stage);
    const uint32_t source_buffer = src_stage.get_base_address();
    const uint32_t dest_buffer = dst_stage.get_base_address();

    uint32_t read_offset = 0;
    uint32_t write_page = write_start_page;
    uint32_t readable = 0;
    uint32_t end_to_write = 0;
    uint32_t writable = dest_page_size_bytes - write_start_offset;

    uint64_t dst_noc_addr = d.get_noc_addr(write_page);
    uint64_t write_offset = (dst_noc_addr & OFFSET_16) + write_start_offset;
    uint64_t begin_write_offset = write_offset;
    constexpr bool can_be_clean = ((source_page_size_bytes % 16) == 0 && (dest_page_size_bytes % 16) == 0);
    uint64_t dst_noc_addr_offset = 0;
    for (uint32_t i = read_start_page; i < read_end_page; i++) {
        // Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i, 0);

        if constexpr (src_aligned_to_64 || ((!src_is_dram) && src_aligned_to_16)) {  // Aligned to 64B, or 16B and L1
            tt::data_movement::common::enhanced_noc_async_read<source_page_size_bytes, false>(
                noc, src_noc_addr, source_buffer, source_page_size_bytes);
            read_offset = 0;
        } else if constexpr (src_is_dram) {  // DRAM but not aligned to 64 (potentially also not aligned to 16)
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
            noc.async_write_barrier();
            if (readable < writable) {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, readable);
                    dst_noc_addr_offset = dst_noc_addr_offset + readable;
                } else {
                    tt::data_movement::common::tt_memmove<false, true, false, dest_page_size_bytes>(
                        noc, dest_buffer + write_offset, source_buffer + read_offset, readable);
                    if (i == read_end_page - 1) {
                        tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                            noc, dest_buffer + begin_write_offset, dst_noc_addr, end_to_write);
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
                    tt::data_movement::common::tt_memmove<false, false, false, dest_page_size_bytes>(
                        noc, dest_buffer + write_offset, source_buffer + read_offset, readable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, dest_buffer + begin_write_offset, dst_noc_addr, dest_page_size_bytes);
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
                    tt::data_movement::common::tt_memmove<false, false, false, dest_page_size_bytes>(
                        noc, dest_buffer + write_offset, source_buffer + read_offset, writable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        noc, dest_buffer + begin_write_offset, dst_noc_addr, dest_page_size_bytes);
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
