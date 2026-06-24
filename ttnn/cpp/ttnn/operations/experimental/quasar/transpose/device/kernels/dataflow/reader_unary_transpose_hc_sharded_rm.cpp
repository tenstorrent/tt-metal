// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 conversion (in place; this kernel is transpose-owned). The device-side NoC + local-copy
// logic is unchanged; only the resource bindings move to the Metal 2.0 namespaces (dfb::/args::).
// cb_in (dfb::cb_in) is the borrowed input shard and cb_out (dfb::cb_out) the borrowed output shard,
// both accessed by L1 address. The variable-length per-core NoC-coordinate / stick-offset lists
// arrive as positional runtime varargs (get_vararg); their counts are recovered from named args.

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
#ifdef USE_SPECIAL_CASE

    bool read_single_h_block_per_core = get_arg(args::read_single_h_block_per_core) == 1;
    uint32_t num_C_blocks_per_core = get_arg(args::num_C_blocks_per_core);
    uint32_t num_sticks_per_shard_core = get_arg(args::num_sticks_per_shard_core);
    uint32_t num_cores_read = get_arg(args::num_cores_read);
    uint32_t read_stick_stride = get_arg(args::read_stick_stride);
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
            uint32_t src_addr = cb_in.get_read_ptr() + get_vararg(core);
            uint32_t l1_write_addr = cb_out.get_write_ptr() + l1_write_offset;

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
                l1_write_addr += write_stick_stride;
            }
            l1_write_offset += stick_size_bytes;
            noc.async_read_barrier();
        }
    } else {
        uint32_t l1_write_addr = cb_out.get_write_ptr();
        uint32_t l1_read_addr = cb_in.get_read_ptr();

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

#else

    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t W_size_bytes = get_arg(args::W_size_bytes);
    constexpr bool row_major = get_arg(args::row_major) == 1;
    constexpr uint32_t num_cores_x = get_arg(args::num_cores_x);
    constexpr uint32_t num_cores_y = get_arg(args::num_cores_y);

    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t curr_c = get_arg(args::curr_c);
    uint32_t curr_h = get_arg(args::curr_h);
    uint32_t curr_n = get_arg(args::curr_n);
    // Varargs (per core): [shard_grid_x_map(num_cores_x) | shard_grid_y_map(num_cores_y)].

    constexpr uint32_t CH = C * H;

    const uint32_t stick_size_bytes = W_size_bytes;

    Noc noc;
    DataflowBuffer cb_in(dfb::cb_in);
    DataflowBuffer cb_out(dfb::cb_out);

    cb_out.reserve_back(num_sticks_per_core);
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core; ++iter) {
        uint32_t shard_id = i_stick / num_sticks_per_core;
        uint32_t stick_id_in_shard = i_stick - (shard_id * num_sticks_per_core);

        uint32_t shard_grid_inner_dim;
        if constexpr (row_major) {
            shard_grid_inner_dim = num_cores_x;
        } else {
            shard_grid_inner_dim = num_cores_y;
        }
        uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
        uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

        uint32_t worker_x_physical, worker_y_physical;
        if constexpr (row_major) {
            worker_x_physical = get_vararg(shard_grid_inner_dim_id);                // shard_grid_x_map[inner]
            worker_y_physical = get_vararg(num_cores_x + shard_grid_outer_dim_id);  // shard_grid_y_map[outer]
        } else {
            worker_x_physical = get_vararg(shard_grid_outer_dim_id);                // shard_grid_x_map[outer]
            worker_y_physical = get_vararg(num_cores_x + shard_grid_inner_dim_id);  // shard_grid_y_map[inner]
        }

        uint32_t l1_read_addr = cb_in.get_read_ptr() + stick_id_in_shard * stick_size_bytes;

        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            stick_size_bytes,
            {.noc_x = worker_x_physical, .noc_y = worker_y_physical, .addr = l1_read_addr},
            {.offset_bytes = 0});
        l1_write_addr += stick_size_bytes;

        curr_c++;
        i_stick += H;
        if (curr_c == C) {  // end of channel dim
            curr_h++;
            curr_c = 0;
            if (curr_h == H) {  // end of H dim
                curr_n++;
                curr_c = 0;
                curr_h = 0;
                i_stick = i_stick - H + 1;
            } else {
                i_stick = i_stick - CH + 1;
            }
        }
    }

    noc.async_read_barrier();
    cb_out.push_back(num_sticks_per_core);

#endif
}
