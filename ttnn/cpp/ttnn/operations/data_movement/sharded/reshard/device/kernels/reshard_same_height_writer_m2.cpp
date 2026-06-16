// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (forked from sharded/device/kernels/dataflow/reshard_same_height_writer.cpp; used only
// by ReshardSameHeightFactory). Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - shard CB c_0   -> dfb::shard_cb (borrowed local shard; read/written by base pointer only -> self-loop)
//   - the remote-buffer base address (legacy RTA slot 3 = a Buffer*) -> the typed TensorAccessor channel
//     (Case-2 bridge: TensorAccessor(ta::remote).get_bank_base_address()).
//   - positional CTAs -> get_arg(args::...); leading scalars stay named RTAs; the variable-length
//     per-segment tail rides as runtime varargs (preserving the legacy packed [write_size, read_offset,
//     bank_id, write_offset] layout the legacy writer read via get_arg_addr(5)).

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
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t total_num_sticks = get_arg(args::total_num_sticks);
    const uint32_t local_stride_bytes = get_arg(args::local_stride_bytes);
    const uint32_t remote_stride_bytes = get_arg(args::remote_stride_bytes);
    const uint32_t num_segments = get_arg(args::num_segments);

    // Per-segment packed tail: [write_size, read_offset, bank_id, write_offset] * num_segments.
    uint32_t vararg_idx = 0;

    DataflowBuffer shard_cb(dfb::shard_cb);
    Noc noc;
    AllocatorBank<bank_type> bank;

    // Remote buffer base address via the typed binding (Case-2 bridge): same offset in every bank.
    auto remote = TensorAccessor(ta::remote);
    const uint32_t base_write_addr = remote.get_bank_base_address();

    uint32_t base_l1_read_addr = shard_cb.get_read_ptr();

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
