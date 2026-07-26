// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

// Writes one prefill slab into a bundle-major KV pool. The compact page table contains one INT32
// physical bundle ID per (slot, logical 5120-token bundle); 32-token subpages within a bundle have
// deterministic offsets and therefore require no per-subpage table entries.
void kernel_main() {
    const uint32_t pool_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(3);

    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(3);
    const uint32_t slot_idx = get_common_arg_val<uint32_t>(8);
    const uint32_t kv_actual_global = get_common_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t input_Ht = get_compile_time_arg_val(3);
    constexpr uint32_t cache_HtWt = get_compile_time_arg_val(4);
    constexpr uint32_t cache_CHtWt = get_compile_time_arg_val(5);
    constexpr uint32_t num_physical_bundles = get_compile_time_arg_val(6);
    constexpr uint32_t num_layers = get_compile_time_arg_val(7);
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t page_table_read_bytes = get_compile_time_arg_val(9);
    constexpr auto pool_args = TensorAccessorArgs<10>();
    constexpr auto page_table_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();

    const uint32_t kv_actual_global_t = kv_actual_global / tile_height;
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_idx = kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;
    const uint32_t update_idxt =
        boundary_slab_idx * chunk_local_t +
        (my_sp_coord < boundary_chip ? chunk_local_t : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    constexpr uint32_t table_entries_per_read = page_table_read_bytes / sizeof(uint32_t);
    constexpr uint32_t invalid_bundle = 0xFFFFFFFF;
    constexpr uint32_t one_page = 1;
    const uint32_t cache_Ht = cache_HtWt / Wt;
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    Noc noc;
    CircularBuffer input_cb(cb_id_out);
    CircularBuffer table_cb(page_table_cb_id);
    const auto pool = TensorAccessor(pool_args, pool_addr);
    const auto page_table = TensorAccessor(page_table_args, page_table_addr);

    table_cb.reserve_back(one_page);
    const uint32_t table_l1 = table_cb.get_write_ptr();
    uint32_t cached_table_segment = invalid_bundle;

    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t input_block = core_blocks_written + page / Wt;
        const uint32_t head = input_block / input_Ht;
        const uint32_t input_row = input_block % input_Ht;
        const uint32_t width_page = page % Wt;

        const uint32_t absolute_local_row = update_idxt + input_row;
        const uint32_t logical_bundle = absolute_local_row / cache_Ht;
        const uint32_t local_row = absolute_local_row % cache_Ht;
        const uint32_t table_segment = logical_bundle / table_entries_per_read;
        if (table_segment != cached_table_segment) {
            noc.async_read(
                page_table,
                CoreLocalMem<uint32_t>(table_l1),
                page_table_read_bytes,
                {.page_id = slot_idx},
                {.offset_bytes = table_segment * page_table_read_bytes});
            noc.async_read_barrier();
            cached_table_segment = table_segment;
        }

        const volatile tt_l1_ptr uint32_t* table_values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
        const uint32_t physical_bundle = table_values[logical_bundle % table_entries_per_read];
        const bool valid_mapping = physical_bundle != invalid_bundle && physical_bundle < num_physical_bundles;

        input_cb.wait_front(one_page);
        if (valid_mapping) {
            const uint32_t physical_batch = physical_bundle * num_layers + layer_idx;
            const uint32_t dst_page = physical_batch * cache_CHtWt + head * cache_HtWt + local_row * Wt + width_page;
            noc.async_write(input_cb, pool, page_bytes, {}, {.page_id = dst_page});
            noc.async_writes_flushed();
        } else {
            // Always drain the input CB, but never derive an address from an invalid entry. Under
            // watcher this ASSERT is fatal and identifies missing allocation without an OOB write.
            ASSERT(valid_mapping);
        }
        input_cb.pop_front(one_page);
    }
    noc.async_write_barrier();
}
