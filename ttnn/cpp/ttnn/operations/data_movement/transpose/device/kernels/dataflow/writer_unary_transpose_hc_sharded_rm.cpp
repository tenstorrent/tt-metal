// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the TransposeHCSharded factory, special-case path). Logic
// UNCHANGED; only the access mechanism moves to named bindings:
//   - input shard CB c_0  -> dfb::src0 (borrowed input shard; read by base pointer only -> self-loop)
//   - output shard CB c_16 -> dfb::out  (borrowed output shard; written by base pointer only -> self-loop)
//   - positional CTAs      -> get_arg(args::...)
//   - the packed variable-length RTA tail (per-core stick offsets + noc x/y) -> runtime varargs. The
//     legacy writer read this tail via get_arg_addr pointer arithmetic; the same packed layout is
//     preserved, now sourced from varargs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
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
    // Packed vararg tail, preserving the legacy get_arg_addr layout:
    //   [0 .. num_cores_read)              : read_stick_offset
    //   [num_cores_read .. 2*num_cores_read) : noc_coord_x
    //   [2*num_cores_read .. 3*num_cores_read) : noc_coord_y
    const uint32_t read_stick_offset_base = 0;
    const uint32_t noc_coord_x_base = num_cores_read;
    const uint32_t noc_coord_y_base = num_cores_read * 2;

    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);

    Noc noc;
    DataflowBuffer cb_in(dfb::src0);
    DataflowBuffer cb_out(dfb::out);

    if (read_single_h_block_per_core) {
        uint32_t write_stick_stride = stick_size_bytes * num_cores_read;

        uint32_t l1_write_offset = 0;
        for (uint32_t core = 0; core < num_cores_read; ++core) {
            uint32_t src_addr =
                cb_in.get_read_ptr() + get_vararg(read_stick_offset_base + core) + src_read_stick_offset;
            uint32_t l1_write_addr = cb_out.get_write_ptr() + l1_write_offset + dst_write_stick_offset;
            uint32_t noc_x = get_vararg(noc_coord_x_base + core);
            uint32_t noc_y = get_vararg(noc_coord_y_base + core);
            for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    dst,
                    stick_size_bytes,
                    {.noc_x = noc_x, .noc_y = noc_y, .addr = src_addr},
                    {.offset_bytes = 0});
                src_addr += read_stick_stride;
                l1_write_addr += write_stick_stride;
            }
            l1_write_offset += stick_size_bytes;
            noc.async_read_barrier();
        }
    } else {
        uint32_t l1_write_addr = cb_out.get_write_ptr() + dst_write_stick_offset;
        uint32_t l1_read_addr = cb_in.get_read_ptr() + src_read_stick_offset;

        for (uint32_t c = 0; c < num_C_blocks_per_core; ++c) {
            for (uint32_t core = 0; core < num_cores_read; ++core) {
                uint32_t src_addr = l1_read_addr + get_vararg(read_stick_offset_base + core);
                uint32_t noc_x = get_vararg(noc_coord_x_base + core);
                uint32_t noc_y = get_vararg(noc_coord_y_base + core);

                noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{}, stick_size_bytes, {.noc_x = noc_x, .noc_y = noc_y, .addr = src_addr});

                for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        UnicastEndpoint{},
                        dst,
                        stick_size_bytes,
                        {.noc_x = noc_x, .noc_y = noc_y, .addr = src_addr},
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
