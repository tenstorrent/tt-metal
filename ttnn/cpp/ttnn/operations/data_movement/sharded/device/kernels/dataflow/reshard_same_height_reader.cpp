// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Named RTAs: total_num_sticks, local_stride_bytes, remote_stride_bytes, num_segments
// Varargs (4 words per segment, positional): read_size, write_offset, bank_id, read_offset
//   (zero-padded by the host to a uniform per-kernel count; only the first num_segments*4 are read.)
void kernel_main() {
    constexpr bool read_from_dram = get_arg(args::interface_with_dram);
    constexpr AllocatorBankType bank_type = read_from_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t total_num_sticks = get_arg(args::total_num_sticks);
    const uint32_t local_stride_bytes = get_arg(args::local_stride_bytes);
    const uint32_t remote_stride_bytes = get_arg(args::remote_stride_bytes);
    const uint32_t num_segments = get_arg(args::num_segments);

    uint32_t args_idx = 0;

    // The shard DFB borrows the local tensor's buffer; it is used here purely as an address source.
    DataflowBuffer shard_dfb(dfb::shard);
    // The remote tensor is walked with raw bank addressing, so only its base address is needed.
    TensorAccessor remote(tensor::remote);
    Noc noc;
    AllocatorBank<bank_type> bank;

    const uint32_t base_read_addr = remote.get_bank_base_address();
    uint32_t base_write_addr = shard_dfb.get_write_ptr();

    for (uint32_t i = 0; i < num_segments; ++i) {
        uint32_t read_size = get_vararg(args_idx++);

        uint32_t write_offset = get_vararg(args_idx++);
        uint32_t l1_write_addr = base_write_addr + write_offset;

        uint32_t bank_id = get_vararg(args_idx++);
        uint32_t read_offset = base_read_addr + get_vararg(args_idx++);

        for (uint32_t j = 0; j < total_num_sticks; ++j) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(bank, dst, read_size, {.bank_id = bank_id, .addr = read_offset}, {.offset_bytes = 0});
            l1_write_addr += local_stride_bytes;
            read_offset += remote_stride_bytes;
        }
    }
    noc.async_read_barrier();
}
