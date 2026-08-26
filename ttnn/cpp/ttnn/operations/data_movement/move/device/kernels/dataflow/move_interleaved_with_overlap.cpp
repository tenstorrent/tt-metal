// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t release_region = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t return_sem_idx = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr uint32_t num_workers = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    constexpr dataflow_kernel_lib::McastArgs<dst_args.next_compile_time_args_offset() + 2, 5> release0_args;
    constexpr dataflow_kernel_lib::
        McastArgs<release0_args.next_compile_time_args_offset(), release0_args.next_runtime_args_offset()>
            release1_args;
    constexpr dataflow_kernel_lib::
        McastArgs<release1_args.next_compile_time_args_offset(), release1_args.next_runtime_args_offset()>
            release2_args;

    Noc noc;
    DataflowBuffer dfb(cb_id);

    Semaphore<> return_sem(return_sem_idx);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = get_tile_size(cb_id);

    const auto src_addrgen = TensorAccessor(src_args, src_addr);
    const auto dst_addrgen = TensorAccessor(dst_args, dst_addr);

    // read a ublock of tiles from src to CB
    dfb.reserve_back(num_tiles);
    uint32_t cb_write_offset = 0;
    for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
        noc.async_read(
            src_addrgen, dfb, tile_bytes, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = cb_write_offset});
        noc.async_read_barrier();
        cb_write_offset += tile_bytes;
    }
    dfb.push_back(num_tiles);

    if (release_region == 3) {
        return_sem.wait(num_workers);
        if constexpr (release0_args.active) {
            release0_args.sender(noc).send_signal();
        }
        if constexpr (release1_args.active) {
            release1_args.sender(noc).send_signal();
        }
        if constexpr (release2_args.active) {
            release2_args.sender(noc).send_signal();
        }
    } else {
        if (release_region == 0) {
            return_sem.up(noc, release0_args.sender_x(), release0_args.sender_y(), 1);
            release0_args.receiver(noc).receive_signal();
        } else if (release_region == 1) {
            return_sem.up(noc, release1_args.sender_x(), release1_args.sender_y(), 1);
            release1_args.receiver(noc).receive_signal();
        } else {
            return_sem.up(noc, release2_args.sender_x(), release2_args.sender_y(), 1);
            release2_args.receiver(noc).receive_signal();
        }
    }

    dfb.wait_front(num_tiles);
    uint32_t cb_read_offset = 0;
    for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
        noc.async_write(
            dfb, dst_addrgen, tile_bytes, {.offset_bytes = cb_read_offset}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        cb_read_offset += tile_bytes;
    }
    dfb.pop_front(num_tiles);
}
