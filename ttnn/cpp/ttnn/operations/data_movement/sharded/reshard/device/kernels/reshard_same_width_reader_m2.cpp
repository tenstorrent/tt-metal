// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (forked from sharded/device/kernels/dataflow/reshard_same_width_reader.cpp; used only
// by ReshardSameWidthFactory). Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - shard CB c_0      -> dfb::shard_cb (borrowed local shard; base-pointer access -> self-loop)
//   - scratch CB c_1    -> dfb::scratch_cb (local scratch, unaligned path; base-pointer access -> self-loop)
//   - remote-buffer base addr (legacy RTA slot 0 = a Buffer*) -> typed TensorAccessor channel
//     (Case-2 bridge: TensorAccessor(ta::remote).get_bank_base_address()).
//   - positional CTAs/RTAs -> get_arg(args::...); the variable-length per-read tail rides as runtime
//     varargs (preserving the legacy packed [bank_id, src_offset, units_to_transfer] layout the legacy
//     reader read via get_arg_addr(3)).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr bool read_from_dram = get_arg(args::read_from_dram);
    // The legacy `unaligned` CTA gate is promoted to a preprocessor define (UNALIGNED) so the
    // conditionally-bound scratch DFB token (dfb::scratch_cb) never enters name lookup on the aligned path.
    constexpr uint32_t unit_size = get_arg(args::unit_size);
    constexpr uint32_t local_unit_size_padded = get_arg(args::local_unit_size_padded);
    constexpr uint32_t remote_unit_size_padded = get_arg(args::remote_unit_size_padded);
    constexpr AllocatorBankType bank_type = read_from_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    // Remote buffer base address via the typed binding (Case-2 bridge): same offset in every bank.
    auto remote = TensorAccessor(ta::remote);
    uint32_t src_addr = remote.get_bank_base_address();
    uint32_t write_offset = get_arg(args::write_offset);
    uint32_t num_reads = get_arg(args::num_reads);
    if (num_reads == 0) {
        return;
    }
    // Per-read packed tail: [bank_id, src_offset, units_to_transfer] * num_reads.
    uint32_t vararg_idx = 0;

    Noc noc;
    AllocatorBank<bank_type> bank;
    DataflowBuffer shard_cb(dfb::shard_cb);

    uint32_t l1_write_addr = shard_cb.get_write_ptr() + write_offset;
#ifdef UNALIGNED
    {
        DataflowBuffer cb_scratch(dfb::scratch_cb);
        uint32_t l1_scratch_write_addr = cb_scratch.get_write_ptr();
        uint32_t l1_scratch_read_addr = cb_scratch.get_read_ptr();
        for (uint32_t i = 0; i < num_reads; ++i) {
            uint32_t bank_id = get_vararg(vararg_idx++);
            uint32_t src_offset = get_vararg(vararg_idx++);
            uint32_t addr = src_addr + src_offset;
            DPRINT("addr: {}\n", addr);
            uint32_t units_to_transfer = get_vararg(vararg_idx++);
            uint32_t read_size = units_to_transfer * remote_unit_size_padded;
            CoreLocalMem<uint32_t> scratch_dst(l1_scratch_write_addr + src_offset);
            noc.async_read(bank, scratch_dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
            noc.async_read_barrier();
            // tt::data_movement::common::print_bf16_pages(
            //     l1_scratch_write_addr + src_offset, remote_unit_size_padded / 2, units_to_transfer);

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
                l1_write_addr += unit_size;
                pad_align_addr += remote_unit_size_padded;
            }
            noc.async_read_barrier();
        }
    }
#else
    {
        for (uint32_t i = 0; i < num_reads; ++i) {
            uint32_t bank_id = get_vararg(vararg_idx++);
            uint32_t addr = src_addr + get_vararg(vararg_idx++);
            uint32_t units_to_transfer = get_vararg(vararg_idx++);
            uint32_t read_size = units_to_transfer * unit_size;
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(bank, dst, read_size, {.bank_id = bank_id, .addr = addr}, {.offset_bytes = 0});
            l1_write_addr += read_size;
        }
        noc.async_read_barrier();
    }
#endif
}
