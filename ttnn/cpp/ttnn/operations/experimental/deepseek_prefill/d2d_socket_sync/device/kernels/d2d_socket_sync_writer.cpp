// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One-shot worker-side producer for D2DStreamServiceSender -- the inverse of
// h2d_socket_sync_writer.cpp. Instead of draining a service-filled backing tensor
// into a fresh output, it fills the sender backing tensor from an input activation
// and signals the persistent sender service that there is data to forward.
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

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// CT-arg layout (must stay in sync with d2d_socket_sync_program_factory.cpp).
constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(2);  // 0 disables the metadata path
// Shared input/backing TensorAccessorArgs (same per-shard spec); metadata accessor,
// when enabled, is packed immediately after.
constexpr auto tensor_accessor_args = TensorAccessorArgs<3>();

// Metadata forward, factored into a template so the metadata-disabled case
// (MetadataSize == 0) is never instantiated -- otherwise the trailing metadata
// TensorAccessorArgs<MetadataAccessorOffset> would static-assert "index out of range"
// whenever the host omits that accessor block (same pattern as the H2D kernel).
template <uint32_t MetadataSize, uint32_t MetadataAccessorOffset>
inline void forward_metadata(uint32_t start_page, uint32_t cb_l1, uint32_t service_noc_x, uint32_t service_noc_y) {
    if constexpr (MetadataSize > 0) {
        if (start_page == 0) {
            const uint32_t metadata_input_addr = get_arg_val<uint32_t>(7);
            const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(8);
            constexpr auto metadata_accessor_args = TensorAccessorArgs<MetadataAccessorOffset>();
            auto metadata_in = TensorAccessor(metadata_accessor_args, metadata_input_addr);
            // Stage the single metadata page through the scratch CB (its data writes
            // are already flushed), then push it to the service core's L1.
            noc_async_read(metadata_in.get_noc_addr(0), cb_l1, MetadataSize);
            noc_async_read_barrier();
            const uint64_t md_noc = get_noc_addr(service_noc_x, service_noc_y, sender_metadata_l1_addr);
            noc_async_write(cb_l1, md_noc, MetadataSize);
            noc_async_write_barrier();
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

    // Input (read source) and backing (write dest) share the per-shard spec: one set
    // of accessor args, two base addresses.
    auto input = TensorAccessor(tensor_accessor_args, input_tensor_addr);
    auto backing = TensorAccessor(tensor_accessor_args, backing_tensor_addr);

    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    // 1. Copy this worker's slice input -> backing through the single-slot CB.
    //    read_barrier before the write; write_barrier before reusing the slot (and, on
    //    the last page, before we signal data_ready in step 3).
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc_async_read(input.get_noc_addr(p), cb_l1, page_size);
        noc_async_read_barrier();
        noc_async_write(cb_l1, backing.get_noc_addr(p), page_size);
        noc_async_write_barrier();
    }

    // 2. (Optional) forward inline metadata to the sender service core.
    forward_metadata<metadata_size_bytes, tensor_accessor_args.next_compile_time_args_offset()>(
        start_page, cb_l1, service_noc_x, service_noc_y);

    // 3. Ack into data_ready_counter -- the service forwards once it has num_workers of
    //    these AND the lease is granted. Return without waiting (the host drives the
    //    lease release/reclaim cadence).
    const uint64_t data_ready_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);
    noc_semaphore_inc(data_ready_noc, 1);
    noc_async_atomic_barrier();
}
