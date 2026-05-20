// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC prefetcher kernel for the DRAM-core mode of ttnn.dram_prefetcher.
//
// Pipeline (ping-pong DMA + NoC push, 2 buffers, single DMA stream):
//   Each K-block of the per-bank weight (kbw * N_per_bank * tile_bytes) is
//   split into `num_dma_chunks_per_block` (M) chunks along the receiver-group
//   dimension. M must divide num_receivers; each chunk holds the contiguous
//   slices of (num_receivers / M) receivers. Stage buffer size = dma_block / M.
//
//   Per K-block we do M push calls, each pushing (num_receivers/M) receivers
//   their full per-K-block slice. Between push calls we *rewind* fifo_wr_ptr
//   to the K-block-start snapshot, so all receivers end up with data at the
//   same fifo offset. The push function advances fifo_wr_ptr by push_chunk_size
//   (= one receiver's full slice per K-block) regardless of how many receivers
//   it iterates, so the net per-K-block advance is exactly push_chunk_size.
//
//   M=1 collapses to the original "one DMA per K-block, push to all receivers
//   in one call" behavior (no rewinding, no iface mutation overhead).
//
//   The DMA ping-pongs across all `num_blocks * M` chunks, giving DMA/NoC
//   overlap both within and across K-blocks — which is what makes M>1 fit
//   DRISC L1 for FF1-class shapes without sacrificing pipelining.
//
//   End-of-stream: remote_cb_sender_barrier, restore NoC mode.

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
    constexpr uint32_t max_chunk_size = get_compile_time_arg_val(4);    // bytes per DMA chunk (doc only)
    constexpr uint32_t stage_buf_addr_a = get_compile_time_arg_val(5);  // ping
    constexpr uint32_t stage_buf_addr_b = get_compile_time_arg_val(6);  // pong
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t pages_sent_l1_addr = get_compile_time_arg_val(8);
    constexpr uint32_t noc_xy_l1_addr = get_compile_time_arg_val(9);
    constexpr uint32_t config_l1_addr = get_compile_time_arg_val(10);
    constexpr uint32_t fifo_size_per_receiver = get_compile_time_arg_val(11);
    constexpr uint32_t receiver_buffer_address = get_compile_time_arg_val(12);
    constexpr uint32_t remote_pages_sent_worker_l1_addr = get_compile_time_arg_val(13);
    constexpr uint32_t num_dma_chunks_per_block = get_compile_time_arg_val(14);  // M
    static_assert(num_dma_chunks_per_block >= 1, "M must be >= 1");
    constexpr uint32_t num_recv_per_chunk = num_receivers / num_dma_chunks_per_block;
    static_assert(
        num_recv_per_chunk * num_dma_chunks_per_block == num_receivers,
        "num_dma_chunks_per_block must divide num_receivers");
    (void)max_chunk_size;

    // ---- Runtime args ----
    //   [0]                 : bank_id (this DRISC's DRAM bank)
    //   [1..1+num_tensors)  : per-tensor bank-local offsets (uint32, into GDDR)
    //   [...]               : per-tensor dma_block_size (bytes read from GDDR per K-block per bank)
    //   [...]               : per-tensor push_page_size (bytes pushed to each receiver per K-block)
    //   [...]               : 2 * num_receivers (noc_x, noc_y per receiver)
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);
    (void)bank_id;

    const uint32_t tensor_offsets_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t tensor_dma_sizes_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t tensor_push_page_sizes_idx = rt_idx;
    rt_idx += num_tensors;

    // ---- One-time L1 setup ----
    volatile tt_l1_ptr uint32_t* noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_l1_addr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        noc_xy_ptr[2 * i] = get_arg_val<uint32_t>(rt_idx++);
        noc_xy_ptr[2 * i + 1] = get_arg_val<uint32_t>(rt_idx++);
    }
    volatile tt_l1_ptr uint32_t* psem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pages_sent_l1_addr);
    const uint32_t psem_uints = (2 * L1_ALIGNMENT * num_receivers) / sizeof(uint32_t);
    for (uint32_t i = 0; i < psem_uints; ++i) {
        psem[i] = 0;
    }
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);
    cfg[0] = 1;
    cfg[1] = num_receivers;
    cfg[2] = receiver_buffer_address;
    cfg[3] = fifo_size_per_receiver;

    RemoteSenderCBInterface& iface = get_remote_sender_cb_interface(remote_cb_id);
    iface.config_ptr = config_l1_addr;
    iface.fifo_start_addr = receiver_buffer_address;
    iface.fifo_wr_ptr = receiver_buffer_address;
    iface.receiver_noc_xy_ptr = noc_xy_l1_addr;
    iface.aligned_pages_sent_ptr = pages_sent_l1_addr;
    iface.num_receivers = num_receivers;
    iface.remote_pages_sent_ptr = remote_pages_sent_worker_l1_addr;

    experimental::drisc_set_stream_mode();

    const uint32_t* tensor_offsets = reinterpret_cast<uint32_t*>(get_arg_addr(tensor_offsets_idx));
    const uint32_t* tensor_dma_sizes = reinterpret_cast<uint32_t*>(get_arg_addr(tensor_dma_sizes_idx));
    const uint32_t* tensor_push_page_sizes = reinterpret_cast<uint32_t*>(get_arg_addr(tensor_push_page_sizes_idx));

    // ---- Main loop ----
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t t = 0; t < num_tensors; ++t) {
            const uint32_t bank_local_base = tensor_offsets[t];
            const uint32_t dma_block_size = tensor_dma_sizes[t];
            const uint32_t push_page_size = tensor_push_page_sizes[t];  // = tpr_full * tile_bytes
            const uint32_t dma_chunk_size = dma_block_size / num_dma_chunks_per_block;
            // push_chunk_size stays at the *full* per-receiver per-K-block slice (push_page_size).
            // With M>1, each push call still pushes push_page_size bytes per receiver, just to a
            // subset of receivers. fifo_wr_ptr advances by push_page_size per push call; we rewind
            // it between chunks within a K-block so the net per-K-block advance is one
            // push_page_size (one fifo page from the sender's perspective).
            experimental::resize_remote_sender_cb_interface<false>(remote_cb_id, push_page_size, noc_index);

            const uint32_t total_chunks = num_blocks * num_dma_chunks_per_block;

            // Prologue: DMA chunk 0 into stage_a.
            uint32_t src_off = bank_local_base;
            experimental::dma_async_read(/*stream=*/0, src_off, stage_buf_addr_a, dma_chunk_size);
            src_off += dma_chunk_size;

            uint32_t fifo_wr_ptr_snapshot = 0;

            for (uint32_t c = 0; c < total_chunks; ++c) {
                const uint32_t chunk_in_block = c % num_dma_chunks_per_block;

                if (chunk_in_block == 0) {
                    // K-block start: reserve space for one push_page_size per receiver (a fifo
                    // page from sender's view = receiver's fifo_page_size if M=K_per_shard).
                    // Reserve_back iterates all num_receivers, so make sure iface has the full
                    // num_receivers + base pointers set.
                    iface.num_receivers = num_receivers;
                    iface.receiver_noc_xy_ptr = noc_xy_l1_addr;
                    iface.aligned_pages_sent_ptr = pages_sent_l1_addr;
                    iface.remote_pages_sent_ptr = remote_pages_sent_worker_l1_addr;
                    experimental::remote_cb_reserve_back(remote_cb_id, 1);
                    fifo_wr_ptr_snapshot = iface.fifo_wr_ptr;
                    // Switch to per-chunk receiver subset for the push calls in this K-block.
                    iface.num_receivers = num_recv_per_chunk;
                }

                const uint32_t stage_buf = (c & 1u) == 0 ? stage_buf_addr_a : stage_buf_addr_b;

                uint32_t outstanding_after_wait = 0;
                if (c + 1 < total_chunks) {
                    const uint32_t next_buf = (c & 1u) == 0 ? stage_buf_addr_b : stage_buf_addr_a;
                    experimental::dma_async_read(/*stream=*/0, src_off, next_buf, dma_chunk_size);
                    src_off += dma_chunk_size;
                    outstanding_after_wait = 1;
                }
                experimental::dma_async_read_wait_n(/*stream=*/0, outstanding_after_wait);

                // Per-chunk push: rewind fifo_wr_ptr to the K-block start so this push writes to
                // the same fifo offset as other chunks within this K-block. Point iface at this
                // chunk's receiver subset.
                iface.fifo_wr_ptr = fifo_wr_ptr_snapshot;
                const uint32_t recv_start = chunk_in_block * num_recv_per_chunk;
                iface.receiver_noc_xy_ptr = noc_xy_l1_addr + recv_start * 2 * sizeof(uint32_t);
                iface.aligned_pages_sent_ptr = pages_sent_l1_addr + recv_start * 2 * L1_ALIGNMENT;
                iface.remote_pages_sent_ptr = remote_pages_sent_worker_l1_addr + recv_start * 2 * L1_ALIGNMENT;

                experimental::remote_cb_push_back_and_write_pages<true>(
                    remote_cb_id,
                    stage_buf,
                    /*num_pages=*/1,
                    /*num_rows=*/1,
                    /*coalesced_num_pages_per_row=*/1,
                    /*coalesced_page_size=*/push_page_size,
                    noc_index);
                noc_async_posted_writes_flushed();
            }

            if (t == num_tensors - 1) {
                // Restore iface to all-receivers for the end-of-stream barrier.
                iface.num_receivers = num_receivers;
                iface.receiver_noc_xy_ptr = noc_xy_l1_addr;
                iface.aligned_pages_sent_ptr = pages_sent_l1_addr;
                iface.remote_pages_sent_ptr = remote_pages_sent_worker_l1_addr;
                experimental::remote_cb_sender_barrier(remote_cb_id);
            }
        }
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    experimental::drisc_set_noc2axi_mode();
}
