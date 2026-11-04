// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    bool read_single_h_block_per_core = get_arg(args::read_single_h_block_per_core) == 1;
    uint32_t num_C_blocks_per_core = get_arg(args::num_C_blocks_per_core);
    uint32_t num_sticks_per_shard_core = get_arg(args::num_sticks_per_shard_core);
    uint32_t num_cores_read = get_arg(args::num_cores_read);
    uint32_t read_stick_stride = get_arg(args::read_stick_stride);
    uint32_t src_read_stick_offset = get_arg(args::src_read_stick_offset);
    uint32_t dst_write_stick_offset = get_arg(args::dst_write_stick_offset);
    // Three per-core arrays of num_cores_read entries ride as runtime varargs, laid out back to
    // back: the stick offset to read from each core, then its NoC x and y coordinates.
    auto read_stick_offset = [&](uint32_t core) { return get_vararg(core); };
    auto noc_coord_x = [&](uint32_t core) { return get_vararg(num_cores_read + core); };
    auto noc_coord_y = [&](uint32_t core) { return get_vararg(num_cores_read * 2 + core); };

    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in0);
    DataflowBuffer dfb_out(dfb::out);

    if (read_single_h_block_per_core) {
        uint32_t write_stick_stride = stick_size_bytes * num_cores_read;

        uint32_t l1_write_offset = 0;
        for (uint32_t core = 0; core < num_cores_read; ++core) {
            uint32_t src_addr = dfb_in.get_read_ptr() + read_stick_offset(core) + src_read_stick_offset;
            uint32_t l1_write_addr = dfb_out.get_write_ptr() + l1_write_offset + dst_write_stick_offset;
            for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    dst,
                    stick_size_bytes,
                    {.noc_x = noc_coord_x(core), .noc_y = noc_coord_y(core), .addr = src_addr},
                    {.offset_bytes = 0});
                src_addr += read_stick_stride;
                l1_write_addr += write_stick_stride;
            }
            l1_write_offset += stick_size_bytes;
            noc.async_read_barrier();
        }
    } else {
        uint32_t l1_write_addr = dfb_out.get_write_ptr() + dst_write_stick_offset;
        uint32_t l1_read_addr = dfb_in.get_read_ptr() + src_read_stick_offset;

        for (uint32_t c = 0; c < num_C_blocks_per_core; ++c) {
            for (uint32_t core = 0; core < num_cores_read; ++core) {
                uint32_t src_addr = l1_read_addr + read_stick_offset(core);

                noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    stick_size_bytes,
                    {.noc_x = noc_coord_x(core), .noc_y = noc_coord_y(core), .addr = src_addr});

                for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        UnicastEndpoint{},
                        dst,
                        stick_size_bytes,
                        {.noc_x = noc_coord_x(core), .noc_y = noc_coord_y(core), .addr = src_addr},
                        {.offset_bytes = 0});
                    src_addr += read_stick_stride;
                    l1_write_addr += stick_size_bytes;
                }
            }
            l1_read_addr += stick_size_bytes;
        }
        noc.async_read_barrier();
    }
}
