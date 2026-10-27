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
#ifdef USE_SPECIAL_CASE

    bool read_single_h_block_per_core = get_arg(args::read_single_h_block_per_core) == 1;
    uint32_t num_C_blocks_per_core = get_arg(args::num_C_blocks_per_core);
    uint32_t num_sticks_per_shard_core = get_arg(args::num_sticks_per_shard_core);
    uint32_t num_cores_read = get_arg(args::num_cores_read);
    uint32_t read_stick_stride = get_arg(args::read_stick_stride);
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
            uint32_t src_addr = dfb_in.get_read_ptr() + read_stick_offset(core);
            uint32_t l1_write_addr = dfb_out.get_write_ptr() + l1_write_offset;

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
                l1_write_addr += write_stick_stride;
            }
            l1_write_offset += stick_size_bytes;
            noc.async_read_barrier();
        }
    } else {
        uint32_t l1_write_addr = dfb_out.get_write_ptr();
        uint32_t l1_read_addr = dfb_in.get_read_ptr();

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

    // The shard grid's physical x coordinates (num_cores_x entries) and y coordinates (num_cores_y
    // entries) ride as runtime varargs, laid out back to back.
    auto shard_grid_x_map = [&](uint32_t i) { return get_vararg(i); };
    auto shard_grid_y_map = [&](uint32_t i) { return get_vararg(num_cores_x + i); };

    constexpr uint32_t CH = C * H;

    const uint32_t stick_size_bytes = W_size_bytes;

    Noc noc;
    DataflowBuffer dfb_in(dfb::in0);
    DataflowBuffer dfb_out(dfb::out);

    dfb_out.reserve_back(num_sticks_per_core);
    uint32_t l1_write_addr = dfb_out.get_write_ptr();

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
            worker_x_physical = shard_grid_x_map(shard_grid_inner_dim_id);
            worker_y_physical = shard_grid_y_map(shard_grid_outer_dim_id);
        } else {
            worker_x_physical = shard_grid_x_map(shard_grid_outer_dim_id);
            worker_y_physical = shard_grid_y_map(shard_grid_inner_dim_id);
        }

        uint32_t l1_read_addr = dfb_in.get_read_ptr() + stick_id_in_shard * stick_size_bytes;

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
    dfb_out.push_back(num_sticks_per_core);

#endif
}
