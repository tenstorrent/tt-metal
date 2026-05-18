// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC prefetcher kernel for the DRAM-core mode of ttnn.dram_prefetcher.
//
// Collapses the BRISC reader + NCRISC writer of the worker-core path into a
// single DRISC kernel:
//   1. DMA-read a block from the local GDDR bank into DRISC L1 (ping-pong
//      between two scratch buffers, stage = block_idx & 1).
//   2. Reserve and push that block to all receivers via the experimental
//      remote_cb_* API (NoC writes from DRISC L1 to each receiver's slice).
//   3. End-of-stream: drain DMA, remote_cb_sender_barrier, restore NoC mode.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_size = get_compile_time_arg_val(4);    // bytes per block per receiver-row
    constexpr uint32_t stage_buf_addr_a = get_compile_time_arg_val(5);  // ping
    constexpr uint32_t stage_buf_addr_b = get_compile_time_arg_val(6);  // pong
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t pages_sent_l1_addr = get_compile_time_arg_val(8);
    constexpr uint32_t noc_xy_l1_addr = get_compile_time_arg_val(9);
    constexpr uint32_t config_l1_addr = get_compile_time_arg_val(10);
    constexpr uint32_t fifo_size_per_receiver = get_compile_time_arg_val(11);
    constexpr uint32_t receiver_buffer_address = get_compile_time_arg_val(12);
    constexpr uint32_t remote_pages_sent_worker_l1_addr = get_compile_time_arg_val(13);

    // ---- Runtime args ----
    // Layout:
    //   [0]                 : bank_id (this DRISC's DRAM bank)
    //   [1..1+num_tensors)  : per-tensor bank-local offsets (uint32, into GDDR)
    //   [...]               : per-tensor block_size (bytes per block)
    //   [...]               : 2 * num_receivers (noc_x, noc_y per receiver)
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);
    (void)bank_id;  // DRISC GDDR is bank-local; address arithmetic does not need it here.

    const uint32_t tensor_offsets_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t tensor_block_sizes_idx = rt_idx;
    rt_idx += num_tensors;

    // ---- One-time L1 setup (mirrors gcb_smoke_sender) ----
    // 1) Receiver noc_xy table.
    volatile tt_l1_ptr uint32_t* noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_l1_addr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        noc_xy_ptr[2 * i] = get_arg_val<uint32_t>(rt_idx++);
        noc_xy_ptr[2 * i + 1] = get_arg_val<uint32_t>(rt_idx++);
    }
    // 2) Zero pages_sent/acked semaphore region.
    volatile tt_l1_ptr uint32_t* psem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pages_sent_l1_addr);
    const uint32_t psem_uints = (2 * L1_ALIGNMENT * num_receivers) / sizeof(uint32_t);
    for (uint32_t i = 0; i < psem_uints; ++i) {
        psem[i] = 0;
    }
    // 3) 4-word mock config (only [3] fifo_size is read by sender path).
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);
    cfg[0] = 1;
    cfg[1] = num_receivers;
    cfg[2] = receiver_buffer_address;
    cfg[3] = fifo_size_per_receiver;

    // 4) RemoteSenderCBInterface.
    RemoteSenderCBInterface& iface = get_remote_sender_cb_interface(remote_cb_id);
    iface.config_ptr = config_l1_addr;
    iface.fifo_start_addr = receiver_buffer_address;
    iface.fifo_wr_ptr = receiver_buffer_address;
    iface.receiver_noc_xy_ptr = noc_xy_l1_addr;
    iface.aligned_pages_sent_ptr = pages_sent_l1_addr;
    iface.num_receivers = num_receivers;
    // DRAM-sender GCB: NOC inc target for pages_sent lives in worker L1, not DRISC L1.
    iface.remote_pages_sent_ptr = remote_pages_sent_worker_l1_addr;

    experimental::drisc_set_stream_mode();

    // Read tensor metadata into locals.
    const uint32_t* tensor_offsets = reinterpret_cast<uint32_t*>(get_arg_addr(tensor_offsets_idx));
    const uint32_t* tensor_block_sizes = reinterpret_cast<uint32_t*>(get_arg_addr(tensor_block_sizes_idx));

    // ---- Main loop ----
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t t = 0; t < num_tensors; ++t) {
            const uint32_t bank_local_base = tensor_offsets[t];
            const uint32_t block_size = tensor_block_sizes[t];
            experimental::resize_remote_sender_cb_interface<false>(remote_cb_id, block_size, noc_index);

            uint32_t src_off = bank_local_base;
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t stage = b & 1u;
                const uint32_t stage_buf = (stage == 0) ? stage_buf_addr_a : stage_buf_addr_b;

                // Issue DMA for this stage.
                experimental::dma_async_read(stage, src_off, stage_buf, block_size);
                src_off += block_size;

                // Wait for THIS stage's DMA to drain before NOC-writing from it.
                experimental::dma_async_read_wait_n(stage, 0);

                // Reserve, write to all receivers, flush.
                experimental::remote_cb_reserve_back(remote_cb_id, 1);
                experimental::remote_cb_push_back_and_write_pages<false>(
                    remote_cb_id, stage_buf, 1, 1, 1, block_size, noc_index);
                noc_async_posted_writes_flushed();
            }

            if (t == num_tensors - 1) {
                experimental::remote_cb_sender_barrier(remote_cb_id);
            }
        }
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    experimental::drisc_set_noc2axi_mode();
}
