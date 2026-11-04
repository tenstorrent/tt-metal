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
// and then, per page of this chip's H2D socket (HOST_PUSH, so the page is already in its L1 FIFO):
//   1. waits until all num_ready participants are done with the previous iteration (and, on the root,
//      until the D2H kernel has pushed the page whose staging half this iteration reuses);
//   2. multicasts the layer's header to every worker core's FE_HDR_ADDR, and the hidden row straight
//      out of the FIFO into every worker core's input replica, and frees the socket page once they
//      have landed;
//   3. multicasts the go words, and sets the reduce senders' go semaphores.
// A page whose layer id is not below num_layers (kFeStopLayer by convention) is the stop: after the
// previous iteration, the D2H kernel is told how many pages it has to push and every RISC its go.
//
// Compile-time args:
//   0: h2d socket config address   1: page_bytes (header + row)   2: row_bytes   3: num_layers
//   4: cb_ctrl (the preloaded headers, then the go words)   5: num_ready   6: is_root
//   7: sem_d2h_done   8: sem_d2h_stop   9: sem_go (the reduce senders')
//   10-14: worker-grid multicast rectangle (NoC 0 start x/y, end x/y) and its core count
//   15: num_senders   16+: TensorAccessorArgs(layer_table)
//
// Runtime args:
//   0: layer_table address   1: input replica address   2+: the senders' NoC xs, then their NoC ys

constexpr uint32_t kFeStopLayer = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t h2d_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_layers = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ctrl_id = get_compile_time_arg_val(4);
    constexpr uint32_t num_ready = get_compile_time_arg_val(5);
    constexpr bool is_root = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t sem_d2h_done_id = get_compile_time_arg_val(7);
    constexpr uint32_t sem_d2h_stop_id = get_compile_time_arg_val(8);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(9);
    constexpr uint32_t mcast_x0 = get_compile_time_arg_val(10);
    constexpr uint32_t mcast_y0 = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_x1 = get_compile_time_arg_val(12);
    constexpr uint32_t mcast_y1 = get_compile_time_arg_val(13);
    constexpr uint32_t mcast_cores = get_compile_time_arg_val(14);
    constexpr uint32_t num_senders = get_compile_time_arg_val(15);
    constexpr auto layer_table_args = TensorAccessorArgs<16>();
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

    SocketReceiverInterface h2d = create_receiver_socket_interface(h2d_config_addr);
    set_receiver_socket_page_size(h2d, page_bytes);

    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_SEM_READY));
    auto* d2h_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_d2h_done_id));
    auto* d2h_stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_d2h_stop_id));
    const uint64_t hdr_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, FE_HDR_ADDR);
    const uint64_t go_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, FE_HDR_ADDR + kHdrBytes);
    const uint64_t row_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, replica_addr);

    auto send_go = [&](uint32_t value) {
        for (uint32_t w = 0; w < kFeGoWords; ++w) {
            go_words[w] = value;
        }
        noc_async_write_multicast(go_l1, go_mcast, kFeGoWords * 4u, mcast_cores);
        for (uint32_t s = 0; s < num_senders; ++s) {
            noc_async_write(go_l1, get_noc_addr(sender_x[s], sender_y[s], get_semaphore(sem_go_id)), 4u);
        }
        noc_async_write_barrier();
    };

    for (uint32_t iter = 0;; ++iter) {
        socket_wait_for_pages(h2d, 1);
        invalidate_l1_cache();
        const uint32_t page = h2d.read_ptr;
        const uint32_t layer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page)[kFeHdrLayer];

        if (iter > 0u) {
            noc_semaphore_wait_min(ready, num_ready * iter);
        }
        if (layer >= num_layers) {
            socket_pop_pages(h2d, 1);
            socket_notify_sender(h2d);
            if constexpr (is_root) {
                *d2h_stop = kFeStopBit | iter;
            }
            send_go(kFeStopBit | (iter + 1u));
            break;
        }
        if constexpr (is_root) {
            if (iter >= 2u) {
                noc_semaphore_wait_min(d2h_done, iter - 1u);
            }
        }

        noc_async_write_multicast(headers_l1 + layer * kHdrBytes, hdr_mcast, kHdrBytes, mcast_cores);
        noc_async_write_multicast(page + kHdrBytes, row_mcast, row_bytes, mcast_cores);
        noc_async_write_barrier();
        socket_pop_pages(h2d, 1);
        socket_notify_sender(h2d);

        send_go(iter + 1u);
    }

    update_socket_config(h2d);
    noc_async_write_barrier();
}
