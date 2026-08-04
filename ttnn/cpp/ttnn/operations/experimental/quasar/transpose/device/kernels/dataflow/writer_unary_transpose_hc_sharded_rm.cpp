// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 conversion (in place; this kernel is transpose-owned). Used only by the special-case
// (split-reader) path of the H<->C row-major sharded transpose. The device-side NoC + local-copy
// logic is unchanged; only the resource bindings move to the Metal 2.0 namespaces (dfb::/args::).
// cb_in (dfb::cb_in) is the borrowed input shard and cb_out (dfb::cb_out) the borrowed output shard,
// both accessed by L1 address. The variable-length per-core NoC-coordinate / stick-offset lists
// arrive as positional runtime varargs (get_vararg); their count is recovered from num_cores_read.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// The CoreLocalMem/UnicastEndpoint NoC-with-state includes disrupt ADL for get_arg (it lives in
// `namespace experimental`), so bring the overload set in explicitly (see the WH sharded RM reader).
using experimental::get_arg;

void kernel_main() {
    bool read_single_h_block_per_core = get_arg(args::read_single_h_block_per_core) == 1;
    uint32_t num_C_blocks_per_core = get_arg(args::num_C_blocks_per_core);
    uint32_t num_sticks_per_shard_core = get_arg(args::num_sticks_per_shard_core);
    uint32_t num_cores_read = get_arg(args::num_cores_read);
    uint32_t read_stick_stride = get_arg(args::read_stick_stride);
    uint32_t src_read_stick_offset = get_arg(args::src_read_stick_offset);
    uint32_t dst_write_stick_offset = get_arg(args::dst_write_stick_offset);
    // Varargs (per core): [read_stick_offset(num_cores_read) | noc_x(num_cores_read) | noc_y(num_cores_read)].

    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);

    Noc noc;
    DataflowBuffer cb_in(dfb::cb_in);
    DataflowBuffer cb_out(dfb::cb_out);

    if (read_single_h_block_per_core) {
        uint32_t write_stick_stride = stick_size_bytes * num_cores_read;

        uint32_t l1_write_offset = 0;
        for (uint32_t core = 0; core < num_cores_read; ++core) {
            uint32_t noc_x = get_vararg(num_cores_read + core);
            uint32_t noc_y = get_vararg(2 * num_cores_read + core);
            uint32_t src_addr = cb_in.get_read_ptr() + get_vararg(core) + src_read_stick_offset;
            uint32_t l1_write_addr = cb_out.get_write_ptr() + l1_write_offset + dst_write_stick_offset;
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
                uint32_t noc_x = get_vararg(num_cores_read + core);
                uint32_t noc_y = get_vararg(2 * num_cores_read + core);
                uint32_t src_addr = l1_read_addr + get_vararg(core);

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
