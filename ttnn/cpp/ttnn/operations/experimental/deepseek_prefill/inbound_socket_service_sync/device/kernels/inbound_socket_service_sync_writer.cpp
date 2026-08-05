// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One-shot worker-side handshake for H2DStreamService.
//
// The backing output and metadata-output base addresses to are runtime args.
// The host registers them as Buffer* BufferBindings
// (KernelDescriptor::emplace_runtime_args), so the device-operation program
// cache can patch the freshly-allocated output address on every dispatch
// without recompiling the program. Everything stable for the service's
// lifetime (sem addr, page counts, metadata addr) stays compile-time.
//
// Per-iteration protocol (worker side):
//   1. Wait on the local data_ready_sem until > 0 (multicast-inc'd by the
//      service core), then reset it to 0.
//   2. (Optional) Snapshot the inline metadata the service core multicast into
//      local L1 to the metadata-output tensor (only the worker owning page 0).
//   3. Copy this worker's [start_page, end_page) slice of the backing tensor
//      into the output tensor, page-by-page, via a single-slot scratch CB.
//   4. Atomic-inc the consumed_counter on the service core. The service core
//      polls for exactly `num_workers` acks per transfer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"  // get_arg_val / get_compile_time_arg_val / noc_semaphore_inc / invalidate_l1_cache
#include "api/dataflow/noc.h"           // Noc
#include "api/dataflow/circular_buffer.h"  // CircularBuffer
#include "api/dataflow/endpoints.h"        // UnicastEndpoint
#include "api/core_local_mem.h"            // CoreLocalMem
#include "api/tensor/tensor_accessor.h"    // TensorAccessor / TensorAccessorArgs
#include "api/tensor/noc_traits.h"         // noc_traits_t for TensorAccessor on the Noc API

// CT-arg layout (must stay in sync with inbound_socket_service_sync_program_factory.cpp).
constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
// Optional metadata path. When `metadata_size_bytes == 0` the metadata read is
// compiled out entirely and the trailing metadata TensorAccessorArgs are not
// consumed
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(4);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(5);

// Metadata snapshot, factored into a template so the metadata-disabled case
// (MetadataSize == 0) is never instantiated. In a plain `if constexpr` inside
// the non-template kernel_main, the discarded branch is still fully checked, so
// TensorAccessorArgs<MetadataAccessorOffset> would static-assert "index out of
// range" whenever the host omits the trailing metadata accessor block.
template <uint32_t MetadataSize, uint32_t MetadataAccessorOffset>
inline void snapshot_metadata(const Noc& noc, uint32_t start_page) {
    if constexpr (MetadataSize > 0) {
        if (start_page == 0) {
            const uint32_t metadata_output_addr = get_arg_val<uint32_t>(7);
            constexpr auto metadata_accessor_args = TensorAccessorArgs<MetadataAccessorOffset>();
            auto metadata_out = TensorAccessor(metadata_accessor_args, metadata_output_addr);
            // Single-page write: the metadata tensor is exactly one page of
            // `MetadataSize` bytes (see the program factory).
            CoreLocalMem<uint32_t> metadata_src(metadata_l1_addr);
            noc.async_write(metadata_src, metadata_out, MetadataSize, {}, {.page_id = 0});
            noc.async_write_barrier();
        }
    }
}

void kernel_main() {
    // RT args (per coord, per worker). Buffer base addresses come first so the
    // host can register them as BufferBindings at fixed positions 0/1 (and 7).
    const uint32_t backing_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t start_page = get_arg_val<uint32_t>(5);
    const uint32_t end_page = get_arg_val<uint32_t>(6);

    // TensorAccessorArgs blocks: backing, output, then (when metadata is
    // enabled) metadata output. The host packs them back-to-back starting at
    // CT-arg index 6.
    constexpr auto backing_accessor_args = TensorAccessorArgs<6>();
    constexpr auto output_accessor_args = TensorAccessorArgs<backing_accessor_args.next_compile_time_args_offset()>();

    auto backing = TensorAccessor(backing_accessor_args, backing_tensor_addr);
    auto output = TensorAccessor(output_accessor_args, output_tensor_addr);

    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);

    // 1. Wait for the service core to signal a fresh transfer, then reset.
    //    The service core multicasts metadata into L1 BEFORE flipping this sem,
    //    so by the time we observe data_ready > 0 the metadata bytes at
    //    `metadata_l1_addr` are already valid.
    CoreLocalMem<volatile uint32_t> data_ready(data_ready_sem_addr);
    while (*data_ready == 0) {
        invalidate_l1_cache();
    }
    *data_ready = 0;

    // 2. (Optional) Snapshot metadata from this worker's local L1 into the
    //    metadata output tensor on DRAM. Gated on `start_page == 0` so a
    //    multi-worker run only emits one write per coord — every worker's L1
    //    holds the same metadata (multicast), so picking one is fine.
    snapshot_metadata<metadata_size_bytes, output_accessor_args.next_compile_time_args_offset()>(noc, start_page);

    // 3. Copy this worker's slice of the backing tensor into the output. Use
    //    the scratch CB as a per-page staging area. read_barrier before write
    //    so we don't issue a write off an unwritten L1 region; write_barrier
    //    before reusing the slot for the next page.
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc.async_read(backing, scratch_cb, page_size, {.page_id = p}, {.offset_bytes = 0});
        noc.async_read_barrier();

        noc.async_write(scratch_cb, output, page_size, {.offset_bytes = 0}, {.page_id = p});
        noc.async_write_barrier();
    }

    // 4. Ack the service core. Exactly one inc per worker per transfer; the
    //    service core's poll checks (cur - last_consumed) == num_workers.
    UnicastEndpoint service;
    const uint64_t consumed_noc =
        service.get_noc_unicast_addr(service_noc_x, service_noc_y, consumed_counter_addr, noc.get_noc_id());
    // Device 2.0 migration: legacy primitive retained
    noc_semaphore_inc(consumed_noc, 1, noc.get_noc_id());
    noc.async_atomic_barrier();
}
