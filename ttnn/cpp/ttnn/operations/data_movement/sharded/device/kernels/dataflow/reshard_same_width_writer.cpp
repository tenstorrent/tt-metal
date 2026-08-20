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

// Named RTAs: read_offset, num_writes
// Varargs (3 words per write, positional): bank_id, dst_offset, units_to_transfer
//   (zero-padded by the host to a uniform per-kernel count; only the first num_writes*3 are read.)
void kernel_main() {
    constexpr bool write_to_dram = get_arg(args::interface_with_dram);
    constexpr uint32_t unit_size = get_arg(args::unit_size);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    uint32_t read_offset = get_arg(args::read_offset);
    uint32_t num_writes = get_arg(args::num_writes);
    if (num_writes == 0) {
        return;
    }
    uint32_t args_idx = 0;

    // The shard DFB borrows the local tensor's buffer; it is used here purely as an address source.
    DataflowBuffer shard_dfb(dfb::shard);
    // The remote tensor is walked with raw bank addressing, so only its base address is needed.
    TensorAccessor remote(tensor::remote);
    Noc noc;
    AllocatorBank<bank_type> bank;

    uint32_t dst_addr = remote.get_bank_base_address();
    uint32_t l1_read_addr = shard_dfb.get_read_ptr() + read_offset;
#ifdef UNALIGNED
    constexpr uint32_t local_unit_size_padded = get_arg(args::local_unit_size_padded);
    constexpr uint32_t remote_unit_size_padded = get_arg(args::remote_unit_size_padded);
    // The local (input) shard stores each row at local_unit_size_padded stride and the remote
    // (output) shard at remote_unit_size_padded stride, and unit_size < either padded stride.
    // A single contiguous write per transfer would pack padding bytes as data and mis-stride
    // rows, so write each row on its own: advance the local source by its padded stride and the
    // remote destination by its padded stride, copying only the unit_size real bytes. Both
    // strides are aligned, so every per-row NOC write hits an aligned address. No scratch is
    // needed here (unlike the reader): the local source is read directly with its L1 address.
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t bank_id = get_vararg(args_idx++);
        uint32_t addr = dst_addr + get_vararg(args_idx++);
        uint32_t units_to_transfer = get_vararg(args_idx++);
        for (uint32_t j = 0; j < units_to_transfer; ++j) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(src, bank, unit_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = addr});
            l1_read_addr += local_unit_size_padded;
            addr += remote_unit_size_padded;
        }
    }
    noc.async_write_barrier();
#else
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t bank_id = get_vararg(args_idx++);
        uint32_t addr = dst_addr + get_vararg(args_idx++);
        uint32_t units_to_transfer = get_vararg(args_idx++);
        uint32_t write_size = units_to_transfer * unit_size;
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, bank, write_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = addr});
        l1_read_addr += write_size;
    }
    noc.async_write_barrier();
#endif
}
