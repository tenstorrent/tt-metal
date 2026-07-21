// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
// 0: address of the D2H socket's sender-side config buffer in L1 on this core.
// 1: aligned page size of the input tensor (must equal the D2H socket's page size).
// 2: scratch CB id used to stage one page between tensor read and PCIe write.
constexpr uint32_t send_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(2);

constexpr uint32_t input_args_cta_idx = 3;
constexpr uint32_t input_args_crta_idx = 0;

// Issues a chunked PCIe NOC write from device L1 into pinned host memory.
// noc_wwrite_with_state can transfer at most NOC_MAX_BURST_SIZE per command, so larger
// pages are split into multiple commands. Caller must flush/barrier afterwards.
FORCE_INLINE void write_page_to_pcie(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t input_base_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);

    // socket_notify_receiver for D2H sockets posts the local bytes_sent counter to the
    // host via a PCIe NOC write, which itself uses noc_wwrite_with_state. We also use
    // noc_wwrite_with_state for the page-data PCIe writes below, so initialize the
    // write command buffer state up front so both hit a hot path.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    SocketSenderInterface sender_socket = create_sender_socket_interface(send_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t data_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;
    uint64_t pcie_data_addr_base =
        (static_cast<uint64_t>(data_addr_hi) << 32) | static_cast<uint64_t>(sender_socket.downstream_fifo_addr);

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_base_addr);

    Noc noc_obj;
    CircularBuffer cb_scratch(scratch_cb_id);

    // The scratch CB is reserved once at kernel startup; nothing else produces/consumes
    // it so we treat it as a single-page L1 staging region.
    uint32_t scratch_addr = cb_scratch.get_write_ptr();

    for (uint32_t page_index = 0; page_index < num_pages; ++page_index) {
        // Block until the host has freed space in the FIFO for one page worth of data.
        socket_reserve_pages(sender_socket, 1);

        // Read this page of the input tensor (L1 or DRAM) into the local L1 scratch.
        noc_obj.async_read(input_addr_gen, CoreLocalMem<uint8_t>(scratch_addr), page_size, {.page_id = page_index}, {});
        noc_obj.async_read_barrier();

        // Push the staged page out to pinned host memory via PCIe.
        write_page_to_pcie(pcie_xy_enc, scratch_addr, pcie_data_addr_base + sender_socket.write_ptr, page_size);
        // Flush so the bytes_sent update below is observed by host after the data has
        // committed to the PCIe ordering domain.
        noc_obj.async_writes_flushed();

        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        invalidate_l1_cache();
    }

    update_socket_config(sender_socket);
    noc_obj.async_write_barrier();
}
