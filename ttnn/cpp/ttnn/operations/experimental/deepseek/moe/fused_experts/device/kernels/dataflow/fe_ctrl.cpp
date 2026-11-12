// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../persistent.h"

// The persistent op's control core (RISCV_0, NoC 0), outside the worker grid (see persistent.h).
//
// It preloads the first kFeHdrWords words of every layer's row of the layer table -- which is laid out
// as the header itself: [layer id, shared gate_up, shared down, router gate, router bias, 0...] --
// and then takes pages from the requesters' input D2D sockets (each a one-page L1 FIFO) in strict round
// robin -- every chip has to run the same row in an iteration, and the chips cannot agree on an arrival
// order, so the order is fixed: requester k's every request takes the next turn after k - 1's. It
// records each iteration's requester in the source ring (after the go words, kFeSourceRing deep), for
// the output kernel, and feeds the row to the two stages that run on this chip, one row apart:
//   1. once the router stage (the router row and the route merge) is done with the previous row, it
//      multicasts the layer's header to the router row's FE_HDR_ADDR, the hidden row into its input
//      replica, and then the router row's go words;
//   2. once the expert stage (the worker grid) is done with the previous row, the reduce stage (the
//      reduce senders, a row behind it) with the one before -- whose half of the senders' staging this
//      row reuses -- and, on the root, the output kernel has pushed the page whose staging half this row
//      reuses: the same header and row to the worker grid, then its go words, the reduce senders' go
//      semaphores and the route merge's sem_expert_go, which holds the row's expert ids until the grid
//      is on it. The socket page is freed then.
// The page is popped without an ack: the requester takes the output row, which every chip's stages have
// had to pass this page to produce, as the ack of all of them.
// A page whose layer id is not below num_layers (kFeStopLayer by convention) closes its requester, which
// then has no more turns. Once every requester is closed, after the previous row has left both stages,
// the output kernel is told how many pages it has to push and every RISC its go.
//
// Compile-time args:
//   0: num_requesters   1: page_bytes (header + row)   2: row_bytes   3: num_layers
//   4: cb_ctrl (the preloaded headers, the go words, then the source ring)   5: num_ready (expert stage)   6: is_root
//   7: sem_out_done   8: sem_out_stop   9: sem_go (the reduce senders')
//   10-14: worker-grid multicast rectangle (NoC 0 start x/y, end x/y) and its core count
//   15: num_senders
//   16-20: router-row multicast rectangle and its core count   21: num_route_ready (router stage)
//   22: sem_route_ready (this core's count of 21)   23-24: route merge core NoC x/y
//   25: sem_expert_go (on the route merge core)
//   26: sem_reduce_ready (this core's count of the reduce senders, num_senders per row)
//   27+: TensorAccessorArgs(layer_table)
//
// Runtime args:
//   0: layer_table address   1: input replica address   2+: the senders' NoC xs, then their NoC ys,
//   then each requester's input socket config address, then the senders' source ring address (kFeSourceRing
//   words on each sender: each iteration's requester, which the senders send its partials to)

constexpr uint32_t kFeStopLayer = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t num_requesters = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_layers = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ctrl_id = get_compile_time_arg_val(4);
    constexpr uint32_t num_ready = get_compile_time_arg_val(5);
    constexpr bool is_root = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t sem_out_done_id = get_compile_time_arg_val(7);
    constexpr uint32_t sem_out_stop_id = get_compile_time_arg_val(8);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(9);
    constexpr uint32_t mcast_x0 = get_compile_time_arg_val(10);
    constexpr uint32_t mcast_y0 = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_x1 = get_compile_time_arg_val(12);
    constexpr uint32_t mcast_y1 = get_compile_time_arg_val(13);
    constexpr uint32_t mcast_cores = get_compile_time_arg_val(14);
    constexpr uint32_t num_senders = get_compile_time_arg_val(15);
    constexpr uint32_t route_x0 = get_compile_time_arg_val(16);
    constexpr uint32_t route_y0 = get_compile_time_arg_val(17);
    constexpr uint32_t route_x1 = get_compile_time_arg_val(18);
    constexpr uint32_t route_y1 = get_compile_time_arg_val(19);
    constexpr uint32_t route_cores = get_compile_time_arg_val(20);
    constexpr uint32_t num_route_ready = get_compile_time_arg_val(21);
    constexpr uint32_t sem_route_ready_id = get_compile_time_arg_val(22);
    constexpr uint32_t merge_x = get_compile_time_arg_val(23);
    constexpr uint32_t merge_y = get_compile_time_arg_val(24);
    constexpr uint32_t sem_expert_go_id = get_compile_time_arg_val(25);
    constexpr uint32_t sem_reduce_ready_id = get_compile_time_arg_val(26);
    constexpr auto layer_table_args = TensorAccessorArgs<27>();
    constexpr uint32_t kHdrBytes = kFeHdrWords * 4u;

    const auto layer_table = TensorAccessor(layer_table_args, get_arg_val<uint32_t>(0));
    const uint32_t replica_addr = get_arg_val<uint32_t>(1);
    tt_l1_ptr uint32_t* sender_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* sender_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + num_senders));

    const uint32_t headers_l1 = get_write_ptr(cb_ctrl_id);
    const uint32_t go_l1 = headers_l1 + num_layers * kHdrBytes;
    auto* go_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_l1);
    for (uint32_t l = 0; l < num_layers; ++l) {
        noc_async_read(layer_table.get_noc_addr(l), headers_l1 + l * kHdrBytes, kHdrBytes);
    }
    noc_async_read_barrier();

    SocketReceiverInterface in[num_requesters];
    for (uint32_t k = 0; k < num_requesters; ++k) {
        in[k] = create_receiver_socket_interface(get_arg_val<uint32_t>(2 + 2 * num_senders + k));
        set_receiver_socket_page_size(in[k], page_bytes);
    }
    auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_l1 + kHdrBytes);
    const uint32_t sender_ring_addr = get_arg_val<uint32_t>(2 + 2 * num_senders + num_requesters);

    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_SEM_READY));
    auto* route_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_route_ready_id));
    auto* reduce_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reduce_ready_id));
    auto* out_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_done_id));
    auto* out_stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_stop_id));
    const uint64_t hdr_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, FE_HDR_ADDR);
    const uint64_t go_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, FE_HDR_ADDR + kHdrBytes);
    const uint64_t row_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, replica_addr);
    const uint64_t route_hdr_mcast = get_noc_multicast_addr(route_x0, route_y0, route_x1, route_y1, FE_HDR_ADDR);
    const uint64_t route_go_mcast =
        get_noc_multicast_addr(route_x0, route_y0, route_x1, route_y1, FE_HDR_ADDR + kHdrBytes);
    const uint64_t route_row_mcast = get_noc_multicast_addr(route_x0, route_y0, route_x1, route_y1, replica_addr);

    auto set_go_words = [&](uint32_t value) {
        for (uint32_t w = 0; w < kFeGoWords; ++w) {
            go_words[w] = value;
        }
    };
    auto send_route_go = [&](uint32_t value) {
        set_go_words(value);
        noc_async_write_multicast(go_l1, route_go_mcast, kFeGoWords * 4u, route_cores);
        noc_async_write_barrier();
    };
    auto send_expert_go = [&](uint32_t value) {
        set_go_words(value);
        noc_async_write_multicast(go_l1, go_mcast, kFeGoWords * 4u, mcast_cores);
        for (uint32_t s = 0; s < num_senders; ++s) {
            noc_async_write(go_l1, get_noc_addr(sender_x[s], sender_y[s], get_semaphore(sem_go_id)), 4u);
        }
        noc_async_write_barrier();
    };

    uint32_t open = (1u << num_requesters) - 1u;
    uint32_t k = num_requesters - 1u;
    for (uint32_t iter = 0;;) {
        do {
            k = k + 1u == num_requesters ? 0u : k + 1u;
        } while (((open >> k) & 1u) == 0u);
        {
            DeviceZoneScopedN("FE_CTRL_PAGE_WAIT");
            socket_wait_for_pages(in[k], 1);
        }
        const uint32_t page = in[k].read_ptr;
        invalidate_l1_cache();
        const uint32_t layer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page)[kFeHdrLayer];

        if (iter > 0u) {
            DeviceZoneScopedN("FE_CTRL_READY_WAIT");
            noc_semaphore_wait_min(route_ready, num_route_ready * iter);
        }
        if (layer >= num_layers) {
            socket_pop_pages(in[k], 1);
            open &= ~(1u << k);
            if (open != 0u) {
                continue;
            }
            noc_semaphore_wait_min(ready, num_ready * iter);
            if constexpr (is_root) {
                *out_stop = kFeStopBit | iter;
            }
            send_route_go(kFeStopBit | (iter + 1u));
            send_expert_go(kFeStopBit | (iter + 1u));
            break;
        }
        source[iter % kFeSourceRing] = k;
        // The senders' copy, which the barrier after the router row's multicasts lands before their go.
        for (uint32_t s = 0; s < num_senders; ++s) {
            noc_async_write(
                reinterpret_cast<uint32_t>(&source[iter % kFeSourceRing]),
                get_noc_addr(sender_x[s], sender_y[s], sender_ring_addr + (iter % kFeSourceRing) * 4u),
                4u);
        }
        {
            DeviceZoneScopedN("FE_CTRL_ROUTE_MCAST");
            noc_async_write_multicast(headers_l1 + layer * kHdrBytes, route_hdr_mcast, kHdrBytes, route_cores);
            noc_async_write_multicast(page + kHdrBytes, route_row_mcast, row_bytes, route_cores);
            noc_async_write_barrier();
            send_route_go(iter + 1u);
        }

        if (iter > 0u) {
            DeviceZoneScopedN("FE_CTRL_EXPERT_WAIT");
            noc_semaphore_wait_min(ready, num_ready * iter);
        }
        if (iter >= 2u) {
            DeviceZoneScopedN("FE_CTRL_REDUCE_WAIT");
            noc_semaphore_wait_min(reduce_ready, num_senders * (iter - 1u));
            if constexpr (is_root) {
                noc_semaphore_wait_min(out_done, iter - 1u);
            }
        }

        {
            DeviceZoneScopedN("FE_CTRL_MCAST");
            noc_async_write_multicast(headers_l1 + layer * kHdrBytes, hdr_mcast, kHdrBytes, mcast_cores);
            noc_async_write_multicast(page + kHdrBytes, row_mcast, row_bytes, mcast_cores);
            noc_async_write_barrier();
            send_expert_go(iter + 1u);
            noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(sem_expert_go_id)), 1);
            noc_async_atomic_barrier();
        }
        socket_pop_pages(in[k], 1);
        ++iter;
    }

    for (uint32_t r = 0; r < num_requesters; ++r) {
        update_socket_config(in[r]);
    }
    noc_async_write_barrier();
}
