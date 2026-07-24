// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One-shot worker producer for the outbound socket services (D2DStreamServiceSender +
// D2HStreamService) -- the inverse of inbound_socket_service_sync_writer.cpp. Fills the
// service backing tensor from an input and acks data_ready (D2D forwards over fabric, D2H
// streams to host). metadata_only mode (D2H, no payload): skips the copy, just forwards
// the record and acks.
//
// The input and backing base addresses are runtime BufferBindings, so the program is
// built once and only the per-dispatch input address is patched on the program-cache
// fast path. The two tensors share the per-shard spec, so a single TensorAccessorArgs
// block describes both (used with the two distinct base addresses).
//
// NON-BLOCKING by design (the model graph's lease cadence depends on it). Per dispatch
// each worker:
//   1. Copy its [start_page, end_page) slice of the input into the sender backing
//      tensor (page-by-page via a single-slot scratch CB). The per-page write barrier
//      also flushes the backing writes before (3).
//   2. (Designated worker -- the one owning page 0, when metadata is enabled) forward
//      the inline metadata blob into the sender service core's L1 metadata buffer,
//      BEFORE acking (the service reads it once it has num_workers acks).
//   3. Atomic-inc data_ready_counter at the service core, then RETURN. The service
//      forwards once it has num_workers acks AND the host grants the fabric lease
//      (release_fabric_links). Waiting on the sender's consumed_sem here would
//      deadlock that order (release is post-dispatch). Back-pressure -- not
//      overwriting the backing before the prior forward drained -- is the host's job:
//      it calls wait_for_fabric_links() on this boundary before the next dispatch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"     // get_arg_val / get_compile_time_arg_val / noc_semaphore_inc
#include "api/dataflow/noc.h"              // Noc
#include "api/dataflow/circular_buffer.h"  // CircularBuffer
#include "api/core_local_mem.h"            // CoreLocalMem
#include "api/dataflow/endpoints.h"        // UnicastEndpoint
#include "api/tensor/tensor_accessor.h"    // TensorAccessor / TensorAccessorArgs
#include "api/tensor/noc_traits.h"         // noc_traits_t for TensorAccessor on the Noc API

// CT-arg layout (must stay in sync with outbound_socket_service_sync_program_factory.cpp).
constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(2);  // 0 disables the metadata path
constexpr uint32_t metadata_only = get_compile_time_arg_val(3);        // 1 no tensor copy
// Shared input/backing TensorAccessorArgs (same per-shard spec); metadata accessor,
// when enabled, is packed immediately after.
constexpr auto tensor_accessor_args = TensorAccessorArgs<4>();

// Metadata forward, factored into a template so the metadata-disabled case
// (MetadataSize == 0) is never instantiated -- otherwise the trailing metadata
// TensorAccessorArgs<MetadataAccessorOffset> would static-assert "index out of range"
// whenever the host omits that accessor block (same pattern as the H2D kernel).
template <uint32_t MetadataSize, uint32_t MetadataAccessorOffset>
inline void forward_metadata(
    const Noc& noc, CircularBuffer& scratch_cb, uint32_t start_page, uint32_t service_noc_x, uint32_t service_noc_y) {
    if constexpr (MetadataSize > 0) {
        if (start_page == 0) {
            const uint32_t metadata_input_addr = get_arg_val<uint32_t>(7);
            const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(8);
            constexpr auto metadata_accessor_args = TensorAccessorArgs<MetadataAccessorOffset>();
            auto metadata_in = TensorAccessor(metadata_accessor_args, metadata_input_addr);
            UnicastEndpoint service;
            // Stage the single metadata page through the scratch CB, then push it to the
            // service core's L1 metadata buffer (a raw unicast L1 dest, not a tensor).
            noc.async_read(metadata_in, scratch_cb, MetadataSize, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            CoreLocalMem<uint32_t> md_src(scratch_cb.get_write_ptr());
            noc.async_write(
                md_src,
                service,
                MetadataSize,
                {},
                {.noc_x = service_noc_x, .noc_y = service_noc_y, .addr = sender_metadata_l1_addr});
            noc.async_write_barrier();
        }
    }
}

void kernel_main() {
    // RT args (per coord, per worker). Base addresses come first so the host can
    // register them as BufferBindings at fixed positions 0/1 (and 7).
    const uint32_t input_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t backing_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t start_page = get_arg_val<uint32_t>(5);
    const uint32_t end_page = get_arg_val<uint32_t>(6);

    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);

    if constexpr (!metadata_only) {
        // Input (read source) and backing (write dest) share the per-shard spec: one set
        // of accessor args, two base addresses.
        auto input = TensorAccessor(tensor_accessor_args, input_tensor_addr);
        auto backing = TensorAccessor(tensor_accessor_args, backing_tensor_addr);

        // 1. Copy this worker's slice input -> backing through the single-slot CB.
        //    read_barrier before the write; write_barrier before reusing the slot (and, on
        //    the last page, before we signal data_ready in step 3).
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc.async_read(input, scratch_cb, page_size, {.page_id = p}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.async_write(scratch_cb, backing, page_size, {.offset_bytes = 0}, {.page_id = p});
            noc.async_write_barrier();
        }
    }

    // 2. (Optional) forward inline metadata to the sender service core.
    forward_metadata<metadata_size_bytes, tensor_accessor_args.next_compile_time_args_offset()>(
        noc, scratch_cb, start_page, service_noc_x, service_noc_y);

    // 3. Ack into data_ready_counter -- the service forwards once it has num_workers of
    //    these AND the lease is granted. Return without waiting (the host drives the
    //    lease release/reclaim cadence).
    UnicastEndpoint service;
    const uint64_t data_ready_noc =
        service.get_noc_unicast_addr(service_noc_x, service_noc_y, data_ready_counter_addr, noc.get_noc_id());
    // Device 2.0 migration: legacy primitive retained
    noc_semaphore_inc(data_ready_noc, 1, noc.get_noc_id());
    noc.async_atomic_barrier();
}
