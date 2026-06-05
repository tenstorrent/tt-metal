// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {

	constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool write_to_dram = get_compile_time_arg_val(1);
    constexpr bool unaligned = get_compile_time_arg_val(2);
    constexpr uint32_t unit_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_unit_size_padded = get_compile_time_arg_val(4);
    constexpr uint32_t remote_unit_size_padded = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch_index = get_compile_time_arg_val(6);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t read_offset = get_arg_val<uint32_t>(1);
    uint32_t num_writes = get_arg_val<uint32_t>(2);
    if (num_writes == 0) {
        return;
    }
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    CircularBuffer shard_cb(shard_cb_id);
    Noc noc;
    AllocatorBank<bank_type> bank;

    uint32_t l1_read_addr = shard_cb.get_read_ptr() + read_offset;
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t bank_id = args[args_idx++];
        uint32_t addr = dst_addr + args[args_idx++];
        uint32_t units_to_transfer = args[args_idx++];
        uint32_t write_size = units_to_transfer * unit_size;
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, bank, write_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = addr});
        l1_read_addr += write_size;
    }
    noc.async_write_barrier();
}
