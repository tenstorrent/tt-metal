// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/paged_kv_utils.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

// Writes this chip's share of the pad window back to the cache: the masked partial tile (from the
// compute out CB) goes back to the boundary seq-tile, and every fully-pad seq-tile is zeroed in place.
//
// The compute kernel pushes Wt out tiles UNCONDITIONALLY (the CB protocol is balanced without a
// reader->compute handoff), so this writer always pops them; it just discards the result on a chip
// with no partial tile. The per-call slot_idx / valid_global come from common args (scalar path) or
// are NoC-read each from its own 1-element uint32 tensor (tensor path: slot_idx tensor at common arg
// 10, valid_global tensor at common arg 11), selected by the HasMeta compile flag.
template <bool HasMeta>
static void run_writer() {
    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t zero_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(2);
    // [3] = has_metadata, [4] = metadata CB index. Cache accessor at <5>; metadata accessor after it.
    constexpr uint32_t meta_cb = get_compile_time_arg_val(4);
    constexpr auto cache_args = TensorAccessorArgs<5>();
    constexpr uint32_t meta_args_offset = cache_args.next_compile_time_args_offset();
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();
    constexpr uint32_t page_bundle_args_offset = HasMeta ? meta_args.next_compile_time_args_offset() : meta_args_offset;
    constexpr auto page_bundle_args = TensorAccessorArgs<page_bundle_args_offset>();
    constexpr uint32_t paged_ct_base = page_bundle_args.next_compile_time_args_offset();
    constexpr bool has_paged_cache = get_compile_time_arg_val(paged_ct_base) != 0;
    constexpr uint32_t page_table_cb = get_compile_time_arg_val(paged_ct_base + 1);
    constexpr uint32_t page_size_rows = get_compile_time_arg_val(paged_ct_base + 2);
    constexpr uint32_t page_num_layers = get_compile_time_arg_val(paged_ct_base + 3);
    constexpr uint32_t page_layer_idx = get_compile_time_arg_val(paged_ct_base + 4);
    constexpr uint32_t page_bundle_count = get_compile_time_arg_val(paged_ct_base + 5);

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_bundle_indices_addr = get_arg_val<uint32_t>(1);

    Noc noc;

    ZeroPadChipWork w;
    if constexpr (HasMeta) {
        // NoC-read slot_idx and valid_global, each element 0 of its own 1-element uint32 tensor:
        // slot_idx tensor address = common arg 10, valid_global tensor address = common arg 11. One
        // accessor (kMetaArgsOffset) is reused for both reads (identical layout).
        const uint32_t slot_idx_addr = get_common_arg_val<uint32_t>(10);
        const uint32_t valid_global_addr = get_common_arg_val<uint32_t>(11);
        CircularBuffer cb_meta(meta_cb);
        cb_meta.reserve_back(1);
        const auto s_slot = TensorAccessor(meta_args, slot_idx_addr);
        noc.async_read(s_slot, cb_meta, 4, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t slot = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
        const auto s_valid = TensorAccessor(meta_args, valid_global_addr);
        noc.async_read(s_valid, cb_meta, 4, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t valid_global = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
        cb_meta.push_back(1);
        w = zero_pad_compute_chip_work(slot, valid_global);
    } else {
        w = zero_pad_compute_chip_work();
    }

    const auto s = TensorAccessor(cache_args, cache_addr, cache_tile_bytes);
    const uint32_t base_page = w.batch_page_base + w.base_local_tile * w.Wt;

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
    const PagedKVAccessor<decltype(s)> paged_cache{
        s, page_table_l1, page_size_rows, page_num_layers, 1, page_layer_idx};

    // UNCONDITIONAL: compute always pushes Wt out tiles -> always consume them. Write the masked
    // partial back only on the chip that owns it; otherwise the tiles are discarded.
    CircularBuffer out(out_cb);
    out.wait_front(w.Wt);
    if (w.count != 0 && w.first_partial != 0) {
        for (uint32_t i = 0; i < w.Wt; ++i) {
            if constexpr (has_paged_cache) {
                const auto cursor = paged_cache.cursor(w.base_local_tile);
                const uint64_t dst_noc_addr =
                    paged_cache.get_shard_row_noc_addr(cursor, (cursor.row_in_bundle * w.Wt + i) * cache_tile_bytes);
                noc_async_write(out.get_read_ptr() + i * cache_tile_bytes, dst_noc_addr, cache_tile_bytes);
            } else {
                noc.async_write(
                    out, s, cache_tile_bytes, {.offset_bytes = i * cache_tile_bytes}, {.page_id = base_page + i});
            }
        }
        noc.async_write_barrier();
    }
    out.pop_front(w.Wt);

    // Zero the fully-pad seq-tiles (those after an optional leading partial). Contiguous-local, so a
    // contiguous page range.
    const uint32_t first_full_tile = w.first_partial ? 1u : 0u;
    if (first_full_tile >= w.count) {
        return;  // nothing to zero (count==0, or only the partial tile on this chip)
    }

    // Zero a one-tile L1 scratch, then stream it to every full pad page. The scratch is never
    // push_back'd, so its read pointer equals the just-zeroed write pointer for the DRAM overload.
    CircularBuffer zero(zero_cb);
    zero.reserve_back(1);
    noc.async_write_zeros(zero, cache_tile_bytes);
    noc.write_zeros_l1_barrier();

    if constexpr (has_paged_cache) {
        for (uint32_t logical_tile = w.base_local_tile + first_full_tile; logical_tile < w.base_local_tile + w.count;
             ++logical_tile) {
            const auto cursor = paged_cache.cursor(logical_tile);
            for (uint32_t i = 0; i < w.Wt; ++i) {
                const uint64_t dst_noc_addr =
                    paged_cache.get_shard_row_noc_addr(cursor, (cursor.row_in_bundle * w.Wt + i) * cache_tile_bytes);
                noc_async_write(zero.get_write_ptr(), dst_noc_addr, cache_tile_bytes);
            }
        }
    } else {
        const uint32_t full_start_page = base_page + first_full_tile * w.Wt;
        const uint32_t full_end_page = base_page + w.count * w.Wt;
        for (uint32_t page = full_start_page; page < full_end_page; ++page) {
            noc.async_write_zeros(s, cache_tile_bytes, {.page_id = page}, zero);
        }
    }
    if constexpr (has_paged_cache) {
        noc.async_write_barrier();
    } else {
        noc.write_zeros_dram_barrier();
    }
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(3);
    run_writer<has_metadata>();
}
