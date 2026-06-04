// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// CB -> cache writer for the per-chip-offset kv-cache update op.
//
// Reads the per-call `slot_idx` and `kv_actual_global` from single-element ROW_MAJOR uint32 device
// tensors (on-device, below) and derives `start_id` from them plus structural common rt-args. Those
// two values are therefore omitted from the program hash — successive users/chunks reuse one cached
// program (per layer), and their buffer addresses are patched on the buffer-binding fast cache-hit
// path. `layer_idx`, `num_layers` and `cluster_axis` stay in the hash (structural).
void kernel_main() {
    // Per-core runtime args (buffers arrive as Buffer* bindings -> addresses).
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot_addr = get_arg_val<uint32_t>(1);
    const uint32_t kv_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_pages = get_arg_val<uint32_t>(3);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(4);

    // Common runtime args (same for all cores on this chip): structural per cached program.
    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(3);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(4);
    const uint32_t Wt = get_common_arg_val<uint32_t>(5);
    const uint32_t cache_HtWt = get_common_arg_val<uint32_t>(6);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t slot_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t kv_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t tile_height = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    constexpr auto slot_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<slot_args.next_compile_time_args_offset()>();

    // Read the single on-device slot_idx datum into a CB. Its address is patched on cache hits; its
    // value is not hashed.
    const auto s_slot = TensorAccessor(slot_args, slot_addr);
    cb_reserve_back(slot_cb_id, 1);
    const uint32_t slot_l1 = get_write_ptr(slot_cb_id);
    noc_async_read_page(0, s_slot, slot_l1);
    noc_async_read_barrier();
    cb_push_back(slot_cb_id, 1);
    const uint32_t slot_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1)[0];

    // Read the single on-device kv_actual_global datum (prior valid global KV length in tokens) and
    // convert to tiles.
    const auto s_kv = TensorAccessor(kv_args, kv_addr);
    cb_reserve_back(kv_cb_id, 1);
    const uint32_t kv_l1 = get_write_ptr(kv_cb_id);
    noc_async_read_page(0, s_kv, kv_l1);
    noc_async_read_barrier();
    cb_push_back(kv_cb_id, 1);
    const uint32_t kv_actual_global_t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_l1)[0] / tile_height;

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
