// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    constexpr uint32_t stick_size_unpadded = get_arg(args::stick_size_unpadded);
    // Buffer's aligned row stride (>= payload when W·E is not 16-aligned); begins_bytes = slice_start[-1] * E.
    constexpr uint32_t src_stride_bytes = get_arg(args::src_stride_bytes);
    constexpr uint32_t dst_stride_bytes = get_arg(args::dst_stride_bytes);
    constexpr uint32_t begins_bytes = get_arg(args::begins_bytes);

    // One coalesced read only when every row is contiguous on both sides and starts at column 0.
    constexpr bool can_coalesce =
        (begins_bytes == 0) && (src_stride_bytes == stick_size_unpadded) && (dst_stride_bytes == stick_size_unpadded);

    const uint32_t num_cores_read = get_arg(args::num_cores_read);

    // The vararg block, in the order the host appends it: one (noc_x, noc_y) pair per source core,
    // then one chunk count per source core, then one (start_id, num_sticks) pair per chunk. How many
    // chunks a source core contributes is itself runtime data, so the pair block is walked with a
    // running offset rather than indexed by source core.
    const uint32_t read_noc_xy_base = 0;
    const uint32_t num_stick_chunks_base = num_cores_read * 2;
    const uint32_t chunk_base = num_cores_read * 3;

    Noc noc;
    const LocalTensorAccessor<uint32_t> src_shard(tensor::src);
    const LocalTensorAccessor<uint32_t> dst_shard(tensor::dst);

    // Raw address needed: a sharded buffer starts at the same SRAM address on every core of its shard
    // grid, so the input's base address locates the input shard on each source core read below. The host
    // passes this address to every core that runs the reader, including cores outside the input's shard
    // grid. Such a core holds no input data at this address, which is fine because the kernel reads at
    // this address only on the source cores, and every source core is in the input's shard grid.
    const uint32_t l1_read_addr =
        static_cast<uint32_t>(local_mem::detail::AddressAccess::address(src_shard.local_mem()));
    // Byte offset into this core's output shard, which the rows fill front to back.
    uint32_t dst_offset = 0;

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
                noc.async_read(
                    UnicastEndpoint{},
                    dst_shard,
                    bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                    {.offset_bytes = dst_offset});
                dst_offset += curr_num_sticks * dst_stride_bytes;
            } else {
                uint32_t src_off = curr_start_id * src_stride_bytes + begins_bytes;
                for (uint32_t s = 0; s < curr_num_sticks; ++s) {
                    noc.async_read(
                        UnicastEndpoint{},
                        dst_shard,
                        stick_size_unpadded,
                        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                        {.offset_bytes = dst_offset});
                    src_off += src_stride_bytes;
                    dst_offset += dst_stride_bytes;
                }
            }
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
}
