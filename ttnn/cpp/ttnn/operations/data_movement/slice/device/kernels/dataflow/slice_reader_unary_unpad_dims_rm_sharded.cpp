// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto stick_size_unpadded = get_arg(args::stick_size_unpadded);
    constexpr auto num_sticks_unpadded = get_arg(args::num_sticks_unpadded);
    // Buffer's aligned row stride (>= payload when W·E is not 16-aligned); begins_bytes = slice_start[-1] * E.
    constexpr auto src_stride_bytes = get_arg(args::src_stride_bytes);
    constexpr auto dst_stride_bytes = get_arg(args::dst_stride_bytes);
    constexpr auto begins_bytes = get_arg(args::begins_bytes);

    // One coalesced read only when every row is contiguous on both sides and starts at column 0.
    constexpr bool can_coalesce =
        (begins_bytes == 0) && (src_stride_bytes == stick_size_unpadded) && (dst_stride_bytes == stick_size_unpadded);

    const uint32_t num_cores_read = get_arg(args::num_cores_read);

    // Everything after the count is one variable-length runtime vararg stream, whose internal layout
    // depends on that count, so its elements are reached by index. Bases, in host push order:
    //   [0, 2*n)      interleaved noc x / y for each source core
    //   [2*n, 3*n)    the number of coalesced stick chunks that source core contributes
    //   [3*n, ...)    (start id, length) for each chunk, in the same core order
    const uint32_t read_noc_xy_base = 0;
    const uint32_t num_stick_chunks_base = num_cores_read * 2;
    const uint32_t chunk_base = num_cores_read * 3;

    Noc noc;
    // Create DataflowBuffers for Device 2.0 API
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    dfb_out.reserve_back(num_sticks_unpadded);
    uint32_t l1_read_addr = dfb_in.get_write_ptr();
    uint32_t l1_write_addr = dfb_out.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(read_noc_xy_base + read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(read_noc_xy_base + read_noc_xy_ptr_offset + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_stick_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset + 1);

            if constexpr (can_coalesce) {
                uint32_t src_off = curr_start_id * src_stride_bytes;
                uint32_t bytes = curr_num_sticks * stick_size_unpadded;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                    {.offset_bytes = 0});
                l1_write_addr += curr_num_sticks * dst_stride_bytes;
            } else {
                uint32_t src_off = curr_start_id * src_stride_bytes + begins_bytes;
                for (uint32_t s = 0; s < curr_num_sticks; ++s) {
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read(
                        UnicastEndpoint{},
                        dst,
                        stick_size_unpadded,
                        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                        {.offset_bytes = 0});
                    src_off += src_stride_bytes;
                    l1_write_addr += dst_stride_bytes;
                }
            }
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
    dfb_out.push_back(num_sticks_unpadded);
}
