// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// CB -> cache writer for the per-chip-offset kv-cache update op.
//
// Derives `start_id` on-device from per-call common rt-args so that
// `kv_actual_global`, `slot_idx`, `layer_idx` can be omitted from the program
// hash — successive chunks with different values reuse the same cached program;
// only rt-args are refreshed via apply_descriptor_runtime_args on the cache-hit
// slow path. `num_layers` and `cluster_axis` stay in the hash (structural).
void kernel_main() {
    // Per-core runtime args.
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
    // per-request fields: kv_actual_global (index 0, tokens) and slot_idx (index 1).
    // Payload layout: [kv_actual_global, slot_idx, dst_slot, reserved] (4 x uint32).
    constexpr uint32_t cb_id_meta = get_compile_time_arg_val(1);
    constexpr uint32_t kMetadataBytes = 16;
    constexpr uint32_t kTileHeight = 32;
    const uint32_t meta_l1 = get_write_ptr(cb_id_meta);
    const InterleavedAddrGen<true> meta_gen{.bank_base_address = metadata_addr, .page_size = kMetadataBytes};
    noc_async_read(get_noc_addr(0, meta_gen), meta_l1, kMetadataBytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    const uint32_t kv_actual_global_t = meta[0] / kTileHeight;
    const uint32_t slot_idx = meta[1];

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

    uint32_t update_idxt;
    if (my_sp_coord < boundary_chip) {
        update_idxt = (boundary_slab_idx + 1) * chunk_local_t;
    } else if (my_sp_coord == boundary_chip) {
        update_idxt = boundary_slab_idx * chunk_local_t + boundary_offset_t;
    } else {
        update_idxt = boundary_slab_idx * chunk_local_t;
    }

    const uint32_t input_Ht = chunk_local_t;
    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
    const uint32_t start_id =
        start_idx + (core_blocks_written / input_Ht) * cache_HtWt + (core_blocks_written % input_Ht) * Wt;

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    // Compile args: [0]=cb_id_out, [1]=cb_id_meta, [2..]=cache TensorAccessorArgs.
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    Noc noc;
    CircularBuffer cb(cb_id_out);

    constexpr uint32_t onepage = 1;
    const auto s = TensorAccessor(dst_args, dst_addr);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
