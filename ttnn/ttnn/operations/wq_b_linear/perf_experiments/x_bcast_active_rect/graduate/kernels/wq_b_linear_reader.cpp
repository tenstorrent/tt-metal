// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// W stream: this core's (tensor_k_tiles x core_n_tiles) slice into a whole-slice resident cb_w_stream.
// Up to w_reads_in_flight K-block batches are outstanding, each under its own read trid; compute is
// released per K-block. The last w_noc1_k_blocks K-blocks are read by the writer (BRISC, NoC1) straight into
// their slots; this kernel stays the CB's only producer and pushes them once the writer's handoff semaphore
// says they landed. A core with a K-slice of a leftover column (regime R4) first reads that slice into
// cb_w_slice under its own trid, so it is resident before compute reaches it.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/wq_b_linear/kernels/wq_b_linear_w_read.hpp"

void kernel_main() {
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t w_reads_in_flight = get_compile_time_arg_val(4);
    constexpr uint32_t w_noc1_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t w_handoff_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_w_slice = get_compile_time_arg_val(7);
    constexpr auto w_args = TensorAccessorArgs<8>();
    constexpr uint32_t slice_trid = w_reads_in_flight + 1;
    static_assert(slice_trid <= NOC_MAX_TRANSACTION_ID);
    constexpr uint32_t num_own_k_blocks = num_k_blocks - w_noc1_k_blocks;
    static_assert(w_reads_in_flight <= num_own_k_blocks);

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_start = get_arg_val<uint32_t>(1);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_stride = get_arg_val<uint32_t>(3);
    const uint32_t slice_n = get_arg_val<uint32_t>(4);
    const uint32_t slice_kb_start = get_arg_val<uint32_t>(5);
    const uint32_t slice_k_blocks = get_arg_val<uint32_t>(6);

    const uint32_t w_page_bytes = get_tile_size(cb_w_stream);
    const auto w = TensorAccessor(w_args, w_addr, w_page_bytes);

    Noc noc;
    CircularBuffer cb_w(cb_w_stream);
    const uint32_t block_tiles = k_block_tiles * core_n_tiles;
    cb_w.reserve_back(num_k_blocks * block_tiles);

    auto trid_for = [](uint32_t b) { return 1 + (b % w_reads_in_flight); };

    // The CB write pointer advances on every push, so a batch lands (b - blocks_pushed) blocks past it.
    auto issue_block = [&](uint32_t b, uint32_t blocks_pushed) {
        dataflow_kernel_lib::set_read_trid(noc, trid_for(b));
        read_w_block(
            noc,
            w,
            cb_w,
            w_page_bytes,
            b,
            (b - blocks_pushed) * block_tiles * w_page_bytes,
            k_block_tiles,
            tensor_n_tiles,
            n_start,
            core_n_tiles,
            n_stride);
    };

    CircularBuffer cb_ws(cb_w_slice);
    if (slice_k_blocks > 0) {
        cb_ws.reserve_back(slice_k_blocks * k_block_tiles);
        dataflow_kernel_lib::set_read_trid(noc, slice_trid);
        for (uint32_t b = 0; b < slice_k_blocks; ++b) {
            read_w_block(
                noc,
                w,
                cb_ws,
                w_page_bytes,
                slice_kb_start + b,
                b * k_block_tiles * w_page_bytes,
                k_block_tiles,
                tensor_n_tiles,
                slice_n,
                1,
                1);
        }
    }

    MaybeDeviceZoneScope("reader_w_stream");
    {
        MaybeDeviceZoneScope("reader_w_first_issue");
        for (uint32_t b = 0; b < w_reads_in_flight; ++b) {
            issue_block(b, 0);
        }
    }
    if (slice_k_blocks > 0) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, slice_trid);
        cb_ws.push_back(slice_k_blocks * k_block_tiles);
    }
    for (uint32_t b = 0; b < num_own_k_blocks; ++b) {
        {
            MaybeDeviceZoneScope("reader_w_barrier");
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid_for(b));
        }
        if (b + w_reads_in_flight < num_own_k_blocks) {
            MaybeDeviceZoneScope("reader_w_issue");
            issue_block(b + w_reads_in_flight, b);
        }
        cb_w.push_back(block_tiles);
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);

    if constexpr (w_noc1_k_blocks > 0) {
        auto* handoff = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(w_handoff_sem_id));
        noc_semaphore_wait(handoff, 1);
        noc_semaphore_set(handoff, 0);
        cb_w.push_back(w_noc1_k_blocks * block_tiles);
    }
}
