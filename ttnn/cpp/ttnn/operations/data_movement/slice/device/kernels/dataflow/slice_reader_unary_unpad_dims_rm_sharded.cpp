// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

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

    // The vararg block, in the order the host appends it: one bank id per source core, then one chunk
    // count per source core, then one (start_id, num_sticks) pair per chunk. How many chunks a source
    // core contributes is itself runtime data, so the pair block is walked with a running offset rather
    // than indexed by source core.
    const uint32_t bank_id_base = 0;
    const uint32_t num_stick_chunks_base = num_cores_read;
    const uint32_t chunk_base = num_cores_read * 2;

    Noc noc;
    // The input is walked by bank: the host gives the bank id of each source core, and the kernel gives
    // the byte offset into the input shard held in that bank.
    const TensorAccessor src(tensor::src);
    const BankView src_bank(src);
    const LocalTensorAccessor<uint32_t> dst_shard(tensor::dst);

    // Byte offset into this core's output shard, which the rows fill front to back.
    uint32_t dst_offset = 0;

    uint32_t chunk_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_bank_id = get_vararg(bank_id_base + curr_core);

        uint32_t curr_core_num_chunks = get_vararg(num_stick_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset + 1);

            if constexpr (can_coalesce) {
                uint32_t src_off = curr_start_id * src_stride_bytes;
                uint32_t bytes = curr_num_sticks * stick_size_unpadded;
                noc.async_read(
                    src_bank,
                    dst_shard,
                    bytes,
                    {.bank_id = src_bank_id, .offset_bytes = src_off},
                    {.offset_bytes = dst_offset});
                dst_offset += curr_num_sticks * dst_stride_bytes;
            } else {
                uint32_t src_off = curr_start_id * src_stride_bytes + begins_bytes;
                for (uint32_t s = 0; s < curr_num_sticks; ++s) {
                    noc.async_read(
                        src_bank,
                        dst_shard,
                        stick_size_unpadded,
                        {.bank_id = src_bank_id, .offset_bytes = src_off},
                        {.offset_bytes = dst_offset});
                    src_off += src_stride_bytes;
                    dst_offset += dst_stride_bytes;
                }
            }
            chunk_ptr_offset += 2;
        }
    }

    noc.async_read_barrier();
}
