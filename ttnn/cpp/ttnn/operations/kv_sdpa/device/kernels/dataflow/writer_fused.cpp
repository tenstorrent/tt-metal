// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// FlashFused writer: generates the reduce/bcast scalars sdpa_standard() needs, then either
//   * (worker, split-KV only) pushes this core's partial (l, m, o) flash state to its head's reducer
//     and signals the reducer's semaphore, or
//   * (reducer) feeds each child's partial into cb_l_in/cb_m_in/cb_out_o for the compute kernel to
//     merge, then drains the normalized output to this core's Q head of the interleaved output.
//
// The partial-state hop is core-to-core within one program, so the child can address the reducer's
// cb_intermed by its OWN local CB address: cb_intermed is declared over the whole core set with an
// identical size, so its L1 offset is the same on every core.
void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(1);
    constexpr uint32_t reducer_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t partial_block_tiles = get_compile_time_arg_val(3);
    // Split-KV role, compile-time per core set (the program instantiates a reducer variant and a
    // worker variant), so the branch below is resolved at compile time.
    constexpr bool is_reducer = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_children = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_head = get_arg_val<uint32_t>(1);
    const uint32_t reducer_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t reducer_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t my_slot = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_10;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_11;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_13;
    constexpr uint32_t cb_partial_out = tt::CBIndex::c_18;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    const uint32_t im_tb = get_tile_size(cb_intermed);
    const uint32_t slot_bytes = partial_block_tiles * im_tb;

    if (!is_reducer) {
        // WORKER: ship (l, m, o) -- staged contiguously by the compute kernel -- into our slot of the
        // reducer's cb_intermed, then bump the reducer's arrival count. No output write; the reducer
        // owns this head's output row.
        cb_wait_front(cb_partial_out, partial_block_tiles);
        const uint32_t src = get_read_ptr(cb_partial_out);
        const uint32_t dst_local = get_write_ptr(cb_intermed) + my_slot * slot_bytes;
        noc_async_write(src, get_noc_addr(reducer_noc_x, reducer_noc_y, dst_local), slot_bytes);
        noc_async_write_barrier();  // the semaphore must not overtake the data
        noc_semaphore_inc(get_noc_addr(reducer_noc_x, reducer_noc_y, get_semaphore(reducer_sem_id)), 1);
        cb_pop_front(cb_partial_out, partial_block_tiles);
        return;
    }

    // REDUCER: wait for every child, then hand them to the compute one at a time. The 1-deep
    // cb_l_in/cb_m_in and DHt-deep cb_out_o backpressure keeps us in lockstep with the merge loop.
    if (num_children > 0) {
        volatile tt_l1_ptr uint32_t* sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(reducer_sem_id));
        noc_semaphore_wait(sem_ptr, num_children);

        const uint32_t im_base = get_read_ptr(cb_intermed);
        for (uint32_t child = 0; child < num_children; ++child) {
            uint32_t src = im_base + child * slot_bytes;
            // Order must match the compute kernel's staging order: l, m, then o. Issue all three
            // reads, then ONE barrier -- three separate barriers would serialize three round trips
            // per child, and the merge chain is already the serial part of the reduction.
            cb_reserve_back(cb_l_in, 1);
            cb_reserve_back(cb_m_in, 1);
            cb_reserve_back(cb_out_o, DHt);
            noc_async_read(get_noc_addr(reducer_noc_x, reducer_noc_y, src), get_write_ptr(cb_l_in), im_tb);
            noc_async_read(get_noc_addr(reducer_noc_x, reducer_noc_y, src + im_tb), get_write_ptr(cb_m_in), im_tb);
            noc_async_read(
                get_noc_addr(reducer_noc_x, reducer_noc_y, src + 2 * im_tb), get_write_ptr(cb_out_o), DHt * im_tb);
            noc_async_read_barrier();
            cb_push_back(cb_l_in, 1);
            cb_push_back(cb_m_in, 1);
            cb_push_back(cb_out_o, DHt);
        }
    }

    const uint32_t o_tb = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, o_tb);
    cb_wait_front(cb_out, DHt);
    uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t d = 0; d < DHt; ++d) {
        noc_async_write_tile(q_head * DHt + d, out_acc, l1);
        l1 += o_tb;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, DHt);
}
