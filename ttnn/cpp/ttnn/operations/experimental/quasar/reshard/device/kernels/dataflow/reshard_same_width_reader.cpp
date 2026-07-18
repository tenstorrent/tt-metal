// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reshard same-width (HEIGHT_SHARDED -> HEIGHT_SHARDED) reader.
//
// Reads units from the remote tensor into the local sharded buffer.
//   - The remote tensor base address comes from LocalTensorAccessor(tensor::remote).get_bank_base_address().
//   - The local sharded buffer base L1 address comes from DataflowBuffer(dfb::shard_cb).get_write_ptr()
//     (the DFB borrows the local tensor's buffer; it is used here purely as an address source).
//   - When UNALIGNED is defined, a scratch DFB (dfb::scratch_cb) stages padded remote reads before
//     they are gathered into the local buffer (local-to-local NoC copy).
//
// Named RTAs: write_offset, num_reads
// Varargs (3 per read, positional): bank_id, src_offset, units_to_transfer
//   (padded by the host to a uniform max; only the first num_reads*3 entries are read.)

#include <stdint.h>
#include "api/tensor/local_tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/debug/ring_buffer.h"  // DEBUG: reshard ebreak bisect (remove after)

void kernel_main() {
    WATCHER_RING_BUFFER_PUSH(0x22DD0001);  // DEBUG: RRD start
    constexpr bool read_from_dram = get_arg(args::interface_with_dram);
    constexpr uint32_t unit_size = get_arg(args::unit_size);
#ifdef UNALIGNED
    constexpr uint32_t remote_unit_size_padded = get_arg(args::remote_unit_size_padded);
#endif
    constexpr AllocatorBankType bank_type = read_from_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t write_offset = get_arg(args::write_offset);
    const uint32_t num_reads = get_arg(args::num_reads);
    if (num_reads == 0) {
        return;
    }

    LocalTensorAccessor<uint32_t> remote(tensor::remote);
    Noc noc;
    AllocatorBank<bank_type> bank;
    DataflowBuffer shard_cb(dfb::shard_cb);

    const uint32_t src_addr = remote.get_bank_base_address();
    uint32_t l1_write_addr = shard_cb.get_write_ptr() + write_offset;
    uint32_t vararg_idx = 0;
#ifdef UNALIGNED
    DataflowBuffer cb_scratch(dfb::scratch_cb);
    uint32_t l1_scratch_write_addr = cb_scratch.get_write_ptr();
    uint32_t l1_scratch_read_addr = cb_scratch.get_read_ptr();
    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = get_vararg(vararg_idx++);
        uint32_t src_offset = get_vararg(vararg_idx++);
        uint32_t addr = src_addr + src_offset;
        uint32_t units_to_transfer = get_vararg(vararg_idx++);
        uint32_t read_size = units_to_transfer * remote_unit_size_padded;
        CoreLocalMem<uint32_t> scratch_dst(l1_scratch_write_addr + src_offset);
        noc.async_read(bank, scratch_dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
        noc.async_read_barrier();

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
            l1_write_addr += unit_size;
            pad_align_addr += remote_unit_size_padded;
        }
        noc.async_read_barrier();
    }
#else
    // DEBUG: reshard ebreak bisect. If PREREAD (with src_addr/l1_write_addr) shows but POSTLOOP doesn't,
    // the trap is inside noc.async_read(bank,...); if PREREAD is missing, setup (get_bank_base_address/
    // get_write_ptr) trapped. Remove after.
    WATCHER_RING_BUFFER_PUSH(0x22DD0002);  // RRD prealigned-loop
    WATCHER_RING_BUFFER_PUSH((uint32_t)src_addr);
    WATCHER_RING_BUFFER_PUSH((uint32_t)l1_write_addr);
    WATCHER_RING_BUFFER_PUSH((uint32_t)num_reads);
    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = get_vararg(vararg_idx++);
        uint32_t addr = src_addr + get_vararg(vararg_idx++);
        uint32_t units_to_transfer = get_vararg(vararg_idx++);
        uint32_t read_size = units_to_transfer * unit_size;
        if (i == 0) {
            WATCHER_RING_BUFFER_PUSH(0x22DD0003);  // RRD first-read
            WATCHER_RING_BUFFER_PUSH((uint32_t)bank_id);
            WATCHER_RING_BUFFER_PUSH((uint32_t)addr);
            WATCHER_RING_BUFFER_PUSH((uint32_t)read_size);
        }
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(bank, dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
        l1_write_addr += read_size;
    }
    WATCHER_RING_BUFFER_PUSH(0x22DD0004);  // RRD postloop
    noc.async_read_barrier();
    WATCHER_RING_BUFFER_PUSH(0x22DD0005);  // RRD end
#endif
}
