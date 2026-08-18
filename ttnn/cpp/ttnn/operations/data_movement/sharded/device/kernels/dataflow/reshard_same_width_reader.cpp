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

// Named RTAs: write_offset, num_reads
// Varargs (3 words per read, positional): bank_id, src_offset, units_to_transfer
//   (zero-padded by the host to a uniform per-kernel count; only the first num_reads*3 are read.)
void kernel_main() {
    constexpr bool read_from_dram = get_arg(args::interface_with_dram);
    constexpr uint32_t unit_size = get_arg(args::unit_size);
    constexpr AllocatorBankType bank_type = read_from_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    uint32_t write_offset = get_arg(args::write_offset);
    uint32_t num_reads = get_arg(args::num_reads);
    if (num_reads == 0) {
        return;
    }
    uint32_t args_idx = 0;

    // The remote tensor is walked with raw bank addressing, so only its base address is needed.
    TensorAccessor remote(tensor::remote);
    Noc noc;
    AllocatorBank<bank_type> bank;
    // The shard DFB borrows the local tensor's buffer; it is used here purely as an address source.
    DataflowBuffer shard_dfb(dfb::shard);

    uint32_t src_addr = remote.get_bank_base_address();
    uint32_t l1_write_addr = shard_dfb.get_write_ptr() + write_offset;
#ifdef UNALIGNED
    constexpr uint32_t local_unit_size_padded = get_arg(args::local_unit_size_padded);
    constexpr uint32_t remote_unit_size_padded = get_arg(args::remote_unit_size_padded);
    DataflowBuffer dfb_scratch(dfb::scratch);
    uint32_t l1_scratch_write_addr = dfb_scratch.get_write_ptr();
    uint32_t l1_scratch_read_addr = dfb_scratch.get_read_ptr();
    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = get_vararg(args_idx++);
        uint32_t src_offset = get_vararg(args_idx++);
        uint32_t addr = src_addr + src_offset;
        uint32_t units_to_transfer = get_vararg(args_idx++);
        uint32_t read_size = units_to_transfer * remote_unit_size_padded;
        CoreLocalMem<uint32_t> scratch_dst(l1_scratch_write_addr + src_offset);
        noc.async_read(bank, scratch_dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // tt::data_movement::common::print_bf16_pages(
        //     l1_scratch_write_addr + src_offset, remote_unit_size_padded / 2, units_to_transfer);

        // Re-stride each row from the remote-aligned scratch layout into the local buffer.
        // Both src (remote_unit_size_padded) and dst (local_unit_size_padded) strides are
        // L1-aligned, so the per-row copy keeps the output shard's aligned page layout
        // (each row padded to local_unit_size_padded) that to_torch expects.
        uint32_t pad_align_addr = l1_scratch_read_addr + src_offset;
        for (uint32_t j = 0; j < units_to_transfer; ++j) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                unit_size,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = pad_align_addr},
                {.offset_bytes = 0});
            // tt::data_movement::common::print_bf16_pages(l1_write_addr, unit_size / 2, 1);
            l1_write_addr += local_unit_size_padded;
            pad_align_addr += remote_unit_size_padded;
        }
        noc.async_read_barrier();
    }
#else
    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = get_vararg(args_idx++);
        uint32_t addr = src_addr + get_vararg(args_idx++);
        uint32_t units_to_transfer = get_vararg(args_idx++);
        uint32_t read_size = units_to_transfer * unit_size;
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(bank, dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
        l1_write_addr += read_size;
    }
    noc.async_read_barrier();
#endif
}
