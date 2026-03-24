// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t tile_height = get_compile_time_arg_val(9);

    const bool has_mcast_first_group = get_arg_val<uint32_t>(0);
    const bool has_mcast_last_group = get_arg_val<uint32_t>(1);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(3);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(4);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(5);
    const uint32_t num_mcast_cores_mid_group = get_arg_val<uint32_t>(6);

    // first mcast group
    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    // last mcast group
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;
    tt_l1_ptr uint32_t* noc_coord_x;
    tt_l1_ptr uint32_t* noc_coord_y;

    // number of cores in mcast groups
    uint32_t num_mcast_cores_first_group;
    uint32_t num_mcast_cores_last_group;

    // first and last group mcast coordinates passed directly in async_write_multicast calls below

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(11);

        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(11);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(11);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));

    } else {
        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(7));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(7 + num_mcast_cores));
    }

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    experimental::Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    reduce_sender_sem.set(VALID);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;

    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_ex(cb_ex_id);
    experimental::CircularBuffer cb_ex_external(cb_ex_external_id);
    experimental::CircularBuffer cb_in0(cb_in0_id);
    experimental::CircularBuffer cb_repack(cb_repack_id);
    experimental::CircularBuffer cb_repack_out(cb_repack_out_id);
    experimental::CircularBuffer cb_out0(cb_out0_id);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    const DataFormat data_format = get_dataformat(cb_ex_partial_id);
    const uint32_t num_bytes_read = datum_size_bytes;

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    experimental::UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    if constexpr (num_mcast_cores > 1) {
        for (uint32_t m = 0; m < num_batch_group; ++m) {
            for (uint32_t n = 0; n < 2; ++n) {
                cb_ex_partial.wait_front(1);

                uint32_t l1_read_addr_ex_par = cb_ex_partial.get_read_ptr();
                uint32_t l1_write_addr_external = cb_ex_external.get_write_ptr();
                experimental::UnicastEndpoint remote_ep;
                noc.async_read(
                    remote_ep,
                    experimental::CoreLocalMem<uint32_t>(l1_write_addr_external),
                    single_tile_size_bytes,
                    {.noc_x = noc_coord_x[0], .noc_y = noc_coord_y[0], .addr = l1_read_addr_ex_par},
                    {});
                l1_write_addr_external += 16;
                noc.async_read_barrier();

                reduce_receiver_sem.wait(num_mcast_cores - 1);
                reduce_receiver_sem.set(0);

                cb_ex_external.reserve_back(1);
                for (uint32_t i = 0; i < num_mcast_cores - 1; ++i) {
                    experimental::UnicastEndpoint remote_ep;
                    noc.async_read(
                        remote_ep,
                        experimental::CoreLocalMem<uint32_t>(l1_write_addr_external),
                        num_bytes_read,
                        {.noc_x = noc_coord_x[i + 1], .noc_y = noc_coord_y[i + 1], .addr = l1_read_addr_ex_par},
                        {});
                    l1_write_addr_external += 16;
                    noc.async_read_barrier();
                }
                cb_ex_external.push_back(1);

                cb_ex.wait_front(1);
                cb_ex_partial.pop_front(1);

                uint32_t l1_read_addr_ex = cb_ex.get_read_ptr();
                experimental::MulticastEndpoint mcast_dst;
                noc.async_write_multicast(
                    experimental::CoreLocalMem<uint32_t>(l1_read_addr_ex),
                    mcast_dst,
                    num_bytes_read,
                    num_mcast_cores_mid_group,
                    {},
                    {.noc_x_start = mcast_dest_noc_start_x,
                     .noc_y_start = mcast_dest_noc_start_y,
                     .noc_x_end = mcast_dest_noc_end_x,
                     .noc_y_end = mcast_dest_noc_end_y,
                     .addr = l1_read_addr_ex},
                    true);
                reduce_sender_sem.set_multicast(
                    noc,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_mcast_cores_mid_group,
                    false);

                if (has_mcast_first_group) {
                    noc.async_write_multicast(
                        experimental::CoreLocalMem<uint32_t>(l1_read_addr_ex),
                        mcast_dst,
                        num_bytes_read,
                        num_mcast_cores_first_group,
                        {},
                        {.noc_x_start = mcast_first_group_dest_noc_start_x,
                         .noc_y_start = mcast_first_group_dest_noc_start_y,
                         .noc_x_end = mcast_first_group_dest_noc_end_x,
                         .noc_y_end = mcast_first_group_dest_noc_end_y,
                         .addr = l1_read_addr_ex},
                        true);
                    reduce_sender_sem.set_multicast(
                        noc,
                        mcast_first_group_dest_noc_start_x,
                        mcast_first_group_dest_noc_start_y,
                        mcast_first_group_dest_noc_end_x,
                        mcast_first_group_dest_noc_end_y,
                        num_mcast_cores_first_group,
                        false);
                }

                if (has_mcast_last_group) {
                    noc.async_write_multicast(
                        experimental::CoreLocalMem<uint32_t>(l1_read_addr_ex),
                        mcast_dst,
                        num_bytes_read,
                        num_mcast_cores_last_group,
                        {},
                        {.noc_x_start = mcast_last_group_dest_noc_start_x,
                         .noc_y_start = mcast_last_group_dest_noc_start_y,
                         .noc_x_end = mcast_last_group_dest_noc_end_x,
                         .noc_y_end = mcast_last_group_dest_noc_end_y,
                         .addr = l1_read_addr_ex},
                        true);
                    reduce_sender_sem.set_multicast(
                        noc,
                        mcast_last_group_dest_noc_start_x,
                        mcast_last_group_dest_noc_start_y,
                        mcast_last_group_dest_noc_end_x,
                        mcast_last_group_dest_noc_end_y,
                        num_mcast_cores_last_group,
                        false);
                }
                noc.async_write_barrier();
                cb_ex.pop_front(1);
            }
        }
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        experimental::UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
