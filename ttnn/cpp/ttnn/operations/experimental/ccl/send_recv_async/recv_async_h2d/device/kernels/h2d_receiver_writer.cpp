// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
// 0: address of the H2D socket's receiver-side config buffer in L1 on this core.
// 1: aligned page size of the output tensor (must equal the H2D socket's page size).
// 2: 1 if the H2D socket is in DEVICE_PULL mode (kernel pulls from pinned host RAM over
//    PCIe), 0 if HOST_PUSH (host has already pushed pages directly into device L1).
constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr bool pull_from_host = get_compile_time_arg_val(2) != 0;

constexpr uint32_t output_args_cta_idx = 3;
constexpr uint32_t output_args_crta_idx = 0;

// Issues a chunked PCIe NOC read from pinned host memory into device L1.
// noc_read_with_state can transfer at most NOC_MAX_BURST_SIZE per command, so larger
// pages are split into multiple commands. Caller must barrier afterwards.
FORCE_INLINE void read_page_from_pcie(uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, chunk);
        src_pcie += chunk;
        dst_l1 += chunk;
        size -= chunk;
    }
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);

    // socket_notify_sender for H2D sockets posts the local bytes_acked counter back to the
    // host via a PCIe NOC write, which itself uses noc_wwrite_with_state. Initialize the
    // write command buffer state up front so the in-loop notifications hit a hot path.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;
    uint64_t pcie_data_addr_base = (static_cast<uint64_t>(read_addr_hi) << 32) | static_cast<uint64_t>(read_addr_lo);

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr);

    for (uint32_t page_index = 0; page_index < num_pages; ++page_index) {
        // Block until the host has signaled (via bytes_sent) that a page of data is ready
        // in the FIFO (HOST_PUSH) or that a page worth of bytes has been reserved in the
        // pinned host buffer for us to pull (DEVICE_PULL).
        socket_wait_for_pages(receiver_socket, 1);

        if constexpr (pull_from_host) {
            read_page_from_pcie(
                pcie_xy_enc,
                pcie_data_addr_base + receiver_socket.read_ptr - receiver_socket.fifo_addr,
                receiver_socket.read_ptr,
                page_size);
            noc_async_read_barrier();
        }

        auto noc_write_addr = output_addr_gen.get_noc_addr(page_index);
        noc_async_write<page_size>(receiver_socket.read_ptr, noc_write_addr, page_size);
        // Flush so the FIFO slot can be released and reused before the write retires globally.
        noc_async_writes_flushed();

        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
        invalidate_l1_cache();
    }

    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
