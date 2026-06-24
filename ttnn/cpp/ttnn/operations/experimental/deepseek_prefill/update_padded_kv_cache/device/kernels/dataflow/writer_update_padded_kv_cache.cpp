// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// CB -> cache writer for the per-chip-offset kv-cache update op.
//
// Reads the per-request `slot_idx` and `kv_actual_global` on-device from the `metadata` DRAM tensor
// (whose address arrives as a common runtime arg) and derives `start_id` from them plus the
// structural common rt-args. Because those two values live in a device tensor they stay off the host
// dispatch path and out of the program hash — successive users/chunks reuse one cached program (per
// layer). `layer_idx`, `num_layers` and `cluster_axis` stay in the hash (structural).
void kernel_main() {
    // Per-core runtime args (buffers arrive as Buffer* bindings -> addresses).
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);

    // Common runtime args (same for all cores on this chip). kv_actual_global and
    // slot_idx are NOT here — they are read on-device from the metadata tensor below.
    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(3);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(4);
    const uint32_t Wt = get_common_arg_val<uint32_t>(5);
    const uint32_t cache_HtWt = get_common_arg_val<uint32_t>(6);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(7);
    const uint32_t metadata_addr = get_common_arg_val<uint32_t>(8);

    // Read the metadata page from DRAM into the L1-scratch CB, then extract the two
    // per-request fields. Canonical payload layout (the runner's h2d_socket_sync
    // packing): [slot_id, actual_start, actual_end] (uint32). slot_idx = index 0;
    // kv_actual_global (the prior valid KV length, in tokens) = actual_start = index 1.
    // Compile args: [0]=cb_id_out, [1]=cb_id_meta, [2..]=cache accessor, then metadata accessor.
    constexpr uint32_t cb_id_meta = get_compile_time_arg_val(1);
    constexpr uint32_t kMetadataBytes = 16;
    constexpr uint32_t kTileHeight = 32;
    constexpr auto cache_args = TensorAccessorArgs<2>();
    constexpr auto meta_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();

    Noc noc;
    CircularBuffer cb_meta(cb_id_meta);
    const auto s_meta = TensorAccessor(meta_args, metadata_addr);
    cb_meta.reserve_back(1);
    noc.async_read(s_meta, cb_meta, kMetadataBytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    CoreLocalMem<volatile uint32_t> meta(cb_meta.get_write_ptr());
    const uint32_t slot_idx = meta[0];
    const uint32_t kv_actual_global_t = meta[1] / kTileHeight;
    cb_meta.push_back(1);

    // Cache linearization: users outer, layers inner.
    const uint32_t batch_idx = slot_idx * num_layers + layer_idx;

    // Derive this chip's tile-row write offset (update_idxt) into its local cache slab from the
    // global valid length kv_actual_global_t. The boundary chip is the one holding the first pad
    // cell; chips before it have already had their pad consumed, so they write into the next slab;
    // the boundary chip writes mid-slab at boundary_offset_t; chips after it write at the current
    // slab base.
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_idx = kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;

    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * chunk_local_t +
        (my_sp_coord < boundary_chip ? chunk_local_t : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    const uint32_t input_Ht = chunk_local_t;
    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
    const uint32_t start_id =
        start_idx + (core_blocks_written / input_Ht) * cache_HtWt + (core_blocks_written % input_Ht) * Wt;

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    CircularBuffer cb(cb_id_out);

    constexpr uint32_t onepage = 1;
    // `cache_args` (cache TensorAccessorArgs) and `noc` are declared above with the metadata read.
    const auto s = TensorAccessor(cache_args, dst_addr);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
