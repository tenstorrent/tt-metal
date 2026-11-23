// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// read_padded_kv_cache: copy the page-rows update_padded_kv_cache stored for one chunk back out of a ROW_MAJOR
// cache. Output row l * chunk_local + j holds cache row local_cache_offset(kv_actual_global) + j of layer
// first_layer + l of the slot; rows past the cache end are skipped (the write never stores them).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/cache_dataflow_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t meta_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t batch_rows = get_compile_time_arg_val(2);
    constexpr auto cache_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t meta_args_offset = out_args.next_compile_time_args_offset();

    const uint32_t linear_coord = kv_cache_dataflow::linear_coord_from_chip_table();
    const uint32_t linear_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local = get_common_arg_val<uint32_t>(2);
    const uint32_t first_layer = get_common_arg_val<uint32_t>(3);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(4);
    const uint32_t cache_rows = get_common_arg_val<uint32_t>(5);
    const uint32_t slot_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t kv_actual_addr = get_common_arg_val<uint32_t>(7);
    const uint32_t cache_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(9);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t first_row = get_arg_val<uint32_t>(1);

    Noc noc;
    CircularBuffer cb_meta(meta_cb_id);
    cb_meta.reserve_back(1);
    const uint32_t slot_idx = kv_cache_dataflow::read_metadata<meta_args_offset>(noc, cb_meta, slot_addr);
    const uint32_t kv_actual_global = kv_cache_dataflow::read_metadata<meta_args_offset>(noc, cb_meta, kv_actual_addr);
    cb_meta.push_back(1);

    // ROW_MAJOR: a page-row is one token row, so the token count is already in page-rows.
    const uint32_t start =
        kv_cache_dataflow::local_cache_offset(kv_actual_global, linear_coord, linear_factor, chunk_local);
    const uint32_t first_entry = slot_idx * num_layers + first_layer;

    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    // The rows CB is this kernel's own staging buffer: rows land at its write pointer and leave from its read
    // pointer, which coincide because nothing is ever pushed or popped.
    CircularBuffer cb(cb_id);
    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    cb.reserve_back(batch_rows);

    const uint32_t end_row = first_row + num_rows;
    for (uint32_t row = first_row; row < end_row; row += batch_rows) {
        const uint32_t count = (end_row - row < batch_rows) ? end_row - row : batch_rows;
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t i = row + k;
            const uint32_t local = start + i % chunk_local;
            if (local < cache_rows) {
                const uint32_t entry = first_entry + i / chunk_local;
                noc.async_read(
                    cache, cb, page_bytes, {.page_id = entry * cache_rows + local}, {.offset_bytes = k * page_bytes});
            }
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t i = row + k;
            if (start + i % chunk_local < cache_rows) {
                noc.async_write(cb, out, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = i});
            }
        }
        noc.async_write_barrier();
    }
}
