// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_pages = get_arg_val<uint32_t>(3);
    uint32_t release_region = get_arg_val<uint32_t>(4);
    uint32_t aligned_page_size = get_arg_val<uint32_t>(5);

    constexpr uint32_t dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t return_sem_idx = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr uint32_t num_workers = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    using Release0Args = dataflow_kernel_lib::McastArgs<dst_args.next_compile_time_args_offset() + 2, 6>;
    using Release1Args = dataflow_kernel_lib::
        McastArgs<Release0Args::next_compile_time_args_offset(), Release0Args::next_runtime_args_offset()>;
    using Release2Args = dataflow_kernel_lib::
        McastArgs<Release1Args::next_compile_time_args_offset(), Release1Args::next_runtime_args_offset()>;

    Noc noc;
    DataflowBuffer dfb(dfb_id);

    const auto src_addrgen = TensorAccessor(src_args, src_addr);
    const auto dst_addrgen = TensorAccessor(dst_args, dst_addr);

    Semaphore<> return_sem(return_sem_idx);

    // read a ublock of tiles from src to CB
    dfb.reserve_back(num_pages);
    uint32_t l1_write_addr = dfb.get_write_ptr();
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(src_addrgen, dst, page_size, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        l1_write_addr += aligned_page_size;
    }
    dfb.push_back(num_pages);

    if (release_region == 3) {
        return_sem.wait(num_workers);
        if constexpr (Release0Args::active) {
            Release0Args().sender(noc).send_signal();
        }
        if constexpr (Release1Args::active) {
            Release1Args().sender(noc).send_signal();
        }
        if constexpr (Release2Args::active) {
            Release2Args().sender(noc).send_signal();
        }
    } else {
        if (release_region == 0) {
            return_sem.up(noc, Release0Args().sender_x(), Release0Args().sender_y(), 1);
            Release0Args().receiver(noc).receive_signal();
        } else if (release_region == 1) {
            return_sem.up(noc, Release1Args().sender_x(), Release1Args().sender_y(), 1);
            Release1Args().receiver(noc).receive_signal();
        } else {
            return_sem.up(noc, Release2Args().sender_x(), Release2Args().sender_y(), 1);
            Release2Args().receiver(noc).receive_signal();
        }
    }

    dfb.wait_front(num_pages);
    uint32_t l1_read_addr = dfb.get_read_ptr();
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, dst_addrgen, page_size, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        l1_read_addr += aligned_page_size;
    }
    dfb.pop_front(num_pages);
}
