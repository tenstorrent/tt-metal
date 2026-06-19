// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (Regime B — wide-W cross-core W-split).
//
// Each core owns one W-shard (Wt_s tiles) of one tile-row group. It:
//   1. reads its input shard once (P1) and its gamma shard once,
//   2. waits for compute's local partial Sum(x^2) (one column-tile),
//   3. all-gathers the K shards' partials over the group's mcast rectangle so
//      every core ends up holding all K partials in cb_partials_gathered,
//   4. compute then sums them to the global Sum(x^2).
//
// The all-gather is a K-round rotating-sender exchange.  Each core is the sender
// in exactly one round (its rank) and a receiver in the others.  Because the
// SAME core flips between sender and receiver across rounds with a SHARED flag
// cell, the Flag-staging pipe would leave a stale "ready" flag on the just-was-
// sender core (SenderPipe::raise_flag_ sets its own cell).  Counter staging
// (monotone, no local self-set) is the correct staging for this rotating pattern
// (mcast_pipe.hpp documents Counter for exactly this census class).  To keep the
// per-round counter accounting exact, the sender forces EXCLUDE_SRC by sending
// with src == dst (a local copy fills its own slot first), so its own counter is
// never self-incremented in its send round.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    // ---- runtime args ----
    const uint32_t my_rank = get_arg_val<uint32_t>(0);  // per-core: rank within the group
    const uint32_t input_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t input_page_base = get_arg_val<uint32_t>(3);
    const uint32_t gamma_page_base = get_arg_val<uint32_t>(4);
    const uint32_t rect_x0 = get_arg_val<uint32_t>(5);
    const uint32_t rect_y0 = get_arg_val<uint32_t>(6);
    const uint32_t rect_x1 = get_arg_val<uint32_t>(7);
    const uint32_t rect_y1 = get_arg_val<uint32_t>(8);
    // sender virtual coords for each round j live at args [9 + 2*j, 9 + 2*j + 1]

    // ---- compile-time args ----
    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial_sumsq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partials_gathered = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_s = get_compile_time_arg_val(5);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(6);
    constexpr uint32_t num_partials = get_compile_time_arg_val(7);  // K
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(9);
    constexpr auto input_args = TensorAccessorArgs<10>();
    [[maybe_unused]] constexpr auto gamma_args =
        TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    using dataflow_kernel_lib::McastRect;
    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReceiverPipe;
    using dataflow_kernel_lib::ReduceDim;
    using dataflow_kernel_lib::SenderPipe;
    using dataflow_kernel_lib::Staging;

    // SUM scaler = 1.0, col-0 (matmul) fill for SUM + REDUCE_ROW.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    const uint32_t tile_bytes = get_tile_size(cb_input_resident);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // gamma shard, read once.
    if constexpr (has_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
        cb_reserve_back(cb_gamma, Wt_s);
        uint32_t l1 = get_write_ptr(cb_gamma);
        for (uint32_t wt = 0; wt < Wt_s; ++wt) {
            noc_async_read_tile(gamma_page_base + wt, gamma_accessor, l1);
            l1 += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt_s);
    }

    // input shard, read once (P1).
    cb_reserve_back(cb_input_resident, Wt_s);
    {
        uint32_t l1 = get_write_ptr(cb_input_resident);
        for (uint32_t wt = 0; wt < Wt_s; ++wt) {
            noc_async_read_tile(input_page_base + wt, input_accessor, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_input_resident, Wt_s);

    // ---- wait for compute's local partial Sum(x^2) ----
    cb_wait_front(cb_partial_sumsq, 1);
    const uint32_t partial_l1 = get_read_ptr(cb_partial_sumsq);

    // ---- all-gather K partials over the group rectangle ----
    cb_reserve_back(cb_partials_gathered, num_partials);
    const uint32_t gathered_base = get_write_ptr(cb_partials_gathered);
    const uint32_t my_slot = gathered_base + my_rank * tile_bytes;

    // Fill my own slot locally (lets the sender use src == dst -> EXCLUDE_SRC).
    noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], partial_l1), my_slot, tile_bytes);
    noc_async_read_barrier();

    Noc noc;
    const McastRect rect{rect_x0, rect_y0, rect_x1, rect_y1};
    SenderPipe<num_partials - 1, data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true>
        sender(noc, rect);
    ReceiverPipe<data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true> receiver(noc);

    for (uint32_t j = 0; j < num_partials; ++j) {
        if (j == my_rank) {
            // src == dst -> no loopback -> EXCLUDE_SRC: write my partial to the OTHER cores'
            // slot j; my own slot j is already filled by the local copy above.
            sender.send(my_slot, my_slot, tile_bytes);
        } else {
            const uint32_t sx = get_arg_val<uint32_t>(9 + 2 * j);
            const uint32_t sy = get_arg_val<uint32_t>(9 + 2 * j + 1);
            receiver.receive(sx, sy);
        }
    }

    cb_push_back(cb_partials_gathered, num_partials);
    cb_pop_front(cb_partial_sumsq, 1);
}
