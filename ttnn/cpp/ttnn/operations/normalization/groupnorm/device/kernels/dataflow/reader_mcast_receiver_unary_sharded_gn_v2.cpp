// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(0);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(1);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(2);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(4);
    constexpr uint32_t tile_height = get_compile_time_arg_val(5);

    constexpr uint32_t operation_ct_args_end = 6;
    constexpr dataflow_kernel_lib::McastArgs<operation_ct_args_end, 0> reduction_mcast_args;

    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;

    Noc noc;
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_repack(dfb_repack_id);
    DataflowBuffer dfb_repack_out(dfb_repack_out_id);
    DataflowBuffer dfb_out0(dfb_out0_id);

    const uint32_t single_tile_size_bytes = get_tile_size(dfb_ex_partial_id);
    const DataFormat data_format = get_dataformat(dfb_ex_partial_id);

    auto reduction_pipe = reduction_mcast_args.receiver(noc);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = dfb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = dfb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        dfb_repack.push_back(per_core_N);
    }
#endif

    for (uint32_t i = 0; i < num_batch_group; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            dfb_ex_partial.wait_front(1);
            dfb_ex_global.reserve_back(1);
            reduction_pipe.receive(dfb_ex_global.get_write_ptr(), single_tile_size_bytes);
            dfb_ex_global.push_back(1);
            dfb_ex_partial.pop_front(1);
        }
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = dfb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = dfb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        dfb_repack_out.pop_front(per_core_N);
    }
#endif
}
