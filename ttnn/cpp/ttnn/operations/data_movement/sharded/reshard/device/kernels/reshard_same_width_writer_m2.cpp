// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (forked from sharded/device/kernels/dataflow/reshard_same_width_writer.cpp; used only
// by ReshardSameWidthFactory). Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - shard CB c_0   -> dfb::shard_cb (borrowed local shard; base-pointer access -> self-loop)
//   - remote-buffer base addr (legacy RTA slot 0 = a Buffer*) -> typed TensorAccessor channel
//     (Case-2 bridge: TensorAccessor(ta::remote).get_bank_base_address()).
//   - positional CTAs/RTAs -> get_arg(args::...); the variable-length per-write tail rides as runtime
//     varargs (preserving the legacy packed [bank_id, dst_offset, units_to_transfer] layout the legacy
//     writer read via get_arg_addr(3)).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr bool write_to_dram = get_arg(args::write_to_dram);
    constexpr bool unaligned = get_arg(args::unaligned);
    constexpr uint32_t unit_size = get_arg(args::unit_size);
    constexpr uint32_t local_unit_size_padded = get_arg(args::local_unit_size_padded);
    constexpr uint32_t remote_unit_size_padded = get_arg(args::remote_unit_size_padded);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    // Remote buffer base address via the typed binding (Case-2 bridge): same offset in every bank.
    auto remote = TensorAccessor(ta::remote);
    uint32_t dst_addr = remote.get_bank_base_address();
    uint32_t read_offset = get_arg(args::read_offset);
    uint32_t num_writes = get_arg(args::num_writes);
    if (num_writes == 0) {
        return;
    }
    // Per-write packed tail: [bank_id, dst_offset, units_to_transfer] * num_writes.
    uint32_t vararg_idx = 0;

    DataflowBuffer shard_cb(dfb::shard_cb);
    Noc noc;
    AllocatorBank<bank_type> bank;

    uint32_t l1_read_addr = shard_cb.get_read_ptr() + read_offset;
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
