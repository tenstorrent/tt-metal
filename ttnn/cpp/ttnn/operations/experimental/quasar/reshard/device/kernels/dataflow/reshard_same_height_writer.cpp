// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reshard same-height (WIDTH_SHARDED -> WIDTH_SHARDED) writer.
//
// Writes contiguous segments from the local sharded buffer out to the remote tensor.
//   - The remote tensor base address comes from LocalTensorAccessor(tensor::remote).get_bank_base_address().
//   - The local sharded buffer base L1 address comes from DataflowBuffer(dfb::shard_cb).get_read_ptr()
//     (the DFB borrows the local tensor's buffer; it is used here purely as an address source).
//
// Named RTAs: total_num_sticks, local_stride_bytes, remote_stride_bytes, num_segments
// Varargs (4 per segment, positional): write_size, read_offset, bank_id, write_offset_rel
//   (padded by the host to a uniform max; only the first num_segments*4 entries are read.)

#include <stdint.h>
#include "api/tensor/local_tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr bool write_to_dram = get_arg(args::interface_with_dram);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t total_num_sticks = get_arg(args::total_num_sticks);
    const uint32_t local_stride_bytes = get_arg(args::local_stride_bytes);
    const uint32_t remote_stride_bytes = get_arg(args::remote_stride_bytes);
    const uint32_t num_segments = get_arg(args::num_segments);

    LocalTensorAccessor<uint32_t> remote(tensor::remote);
    DataflowBuffer shard_cb(dfb::shard_cb);
    Noc noc;
    AllocatorBank<bank_type> bank;

    const uint32_t base_write_addr = remote.get_bank_base_address();
    const uint32_t base_l1_read_addr = shard_cb.get_read_ptr();

    uint32_t vararg_idx = 0;
    for (uint32_t i = 0; i < num_segments; ++i) {
        uint32_t write_size = get_vararg(vararg_idx++);

        uint32_t read_offset = get_vararg(vararg_idx++);
        uint32_t l1_read_addr = base_l1_read_addr + read_offset;

        uint32_t bank_id = get_vararg(vararg_idx++);
        uint32_t write_offset = base_write_addr + get_vararg(vararg_idx++);

        for (uint32_t j = 0; j < total_num_sticks; ++j) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(src, bank, write_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = write_offset});
            l1_read_addr += local_stride_bytes;
            write_offset += remote_stride_bytes;
        }
    }
    noc.async_write_barrier();
}
