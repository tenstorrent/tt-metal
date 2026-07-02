// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reshard same-width (HEIGHT_SHARDED -> HEIGHT_SHARDED) writer.
//
// Writes units from the local sharded buffer out to the remote tensor.
//   - The remote tensor base address comes from TensorAccessor(tensor::remote).get_bank_base_address().
//   - The local sharded buffer base L1 address comes from DataflowBuffer(dfb::shard_cb).get_read_ptr()
//     (the DFB borrows the local tensor's buffer; it is used here purely as an address source).
//
// Named RTAs: read_offset, num_writes
// Varargs (3 per write, positional): bank_id, dst_offset, units_to_transfer
//   (padded by the host to a uniform max; only the first num_writes*3 entries are read.)

#include <stdint.h>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr bool write_to_dram = get_arg(args::interface_with_dram);
    constexpr uint32_t unit_size = get_arg(args::unit_size);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t read_offset = get_arg(args::read_offset);
    const uint32_t num_writes = get_arg(args::num_writes);
    if (num_writes == 0) {
        return;
    }

    TensorAccessor remote(tensor::remote);
    DataflowBuffer shard_cb(dfb::shard_cb);
    Noc noc;
    AllocatorBank<bank_type> bank;

    const uint32_t dst_addr = remote.get_bank_base_address();
    uint32_t l1_read_addr = shard_cb.get_read_ptr() + read_offset;
    uint32_t vararg_idx = 0;
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t bank_id = get_vararg(vararg_idx++);
        uint32_t addr = dst_addr + get_vararg(vararg_idx++);
        uint32_t units_to_transfer = get_vararg(vararg_idx++);
        uint32_t write_size = units_to_transfer * unit_size;
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, bank, write_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = addr});
        l1_read_addr += write_size;
    }
    noc.async_write_barrier();
}
