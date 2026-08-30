// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/paged_kv_utils.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

// Dataflow-only ROW_MAJOR pad cleanup. A cache page is one token row (including aligned row padding),
// so stream an all-zero L1 row to exactly [valid_global, ceil_pad(valid_global)). This works for BF16
// and FP8_E4M3 because the unpack/compute engine never touches the payload.
void kernel_main() {
    constexpr uint32_t zero_cb = get_compile_time_arg_val(0);
    constexpr uint32_t row_page_bytes = get_compile_time_arg_val(1);
    constexpr auto cache_args = TensorAccessorArgs<2>();
    constexpr auto page_bundle_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t paged_ct_base = page_bundle_args.next_compile_time_args_offset();
    constexpr bool has_paged_cache = get_compile_time_arg_val(paged_ct_base) != 0;
    constexpr uint32_t page_table_cb = get_compile_time_arg_val(paged_ct_base + 1);
    constexpr uint32_t page_size_rows = get_compile_time_arg_val(paged_ct_base + 2);
    constexpr uint32_t page_num_layers = get_compile_time_arg_val(paged_ct_base + 3);
    constexpr uint32_t page_layer_idx = get_compile_time_arg_val(paged_ct_base + 4);
    constexpr uint32_t page_bundle_count = get_compile_time_arg_val(paged_ct_base + 5);

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_bundle_indices_addr = get_arg_val<uint32_t>(1);
    const ZeroPadRowMajorChipWork w = zero_pad_compute_row_major_chip_work();
    if (w.count == 0) {
        return;
    }

    const auto cache = TensorAccessor(cache_args, cache_addr, row_page_bytes);
    CircularBuffer zero(zero_cb);
    zero.reserve_back(1);

    Noc noc;
    uint32_t page_table_l1 = 0;
    if constexpr (has_paged_cache) {
        CircularBuffer table_cb(page_table_cb);
        page_table_l1 = table_cb.get_write_ptr();
        const auto table = TensorAccessor(page_bundle_args, page_bundle_indices_addr);
        noc.async_read(
            table, CoreLocalMem<uint16_t>(page_table_l1), page_bundle_count * sizeof(uint16_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        invalidate_l1_cache();
    }
    const PagedKVAccessor<decltype(cache)> paged_cache{
        cache, page_table_l1, page_size_rows, page_num_layers, 1, page_layer_idx};

    noc.async_write_zeros(zero, row_page_bytes);
    noc.write_zeros_l1_barrier();

    if constexpr (has_paged_cache) {
        for (uint32_t logical_row = w.base_local_row; logical_row < w.base_local_row + w.count; ++logical_row) {
            const auto cursor = paged_cache.cursor(logical_row);
            const uint64_t dst_noc_addr =
                paged_cache.get_shard_row_noc_addr(cursor, cursor.row_in_bundle * row_page_bytes);
            noc_async_write(zero.get_write_ptr(), dst_noc_addr, row_page_bytes);
        }
        noc.async_write_barrier();
    } else {
        const uint32_t first_page = w.batch_page_base + w.base_local_row;
        const uint32_t last_page = first_page + w.count;
        for (uint32_t page = first_page; page < last_page; ++page) {
            noc.async_write_zeros(cache, row_page_bytes, {.page_id = page}, zero);
        }
        noc.write_zeros_dram_barrier();
    }
}
