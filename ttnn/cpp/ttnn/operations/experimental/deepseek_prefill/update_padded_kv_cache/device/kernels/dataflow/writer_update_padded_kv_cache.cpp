// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/paged_kv_utils.hpp"

// CB -> cache writer for the per-chip-offset kv-cache update op.
//
// The per-request `slot_idx` and `kv_actual_global` reach the kernel one of two ways, selected by the
// `has_metadata` compile-time flag (the op sets it from whether the metadata tensors were supplied):
//   - metadata path: read on-device from two 1-element uint32 DRAM tensors (raw addresses in common
//     args 8 and 9) -> slot_idx = slot_idx tensor's element [0], kv_actual_global (tokens) =
//     kv_actual_global tensor's element [0]. The values stay off the host dispatch path, so the op is
//     traceable and one cached program per layer is reused across users/chunks.
//   - scalar path: read from common runtime args 8/9 (patched on cache hits by the op's
//     override_runtime_arguments). Kept out of the program hash the same way.
// `layer_idx`, `num_layers` and `cluster_axis` stay in the hash (structural) in both paths.
//
// Optional `valid_global` (end of the chunk's REAL tokens) arrives the same two ways in common arg 10.
// Set, the chip writes only the page-rows holding real tokens; unset, the whole padded slab.
//
// Compile args: [0]=cb_id_out, [1]=has_metadata, [2]=cb_id_meta, [3]=tile_height, [4]=has_valid,
// then the cache and page-table accessors, six paged-cache configuration values, and (metadata path only) one metadata
// accessor. The two 1-element metadata tensors share an identical layout, so the same accessor serves
// both reads. tile_height divides KV tokens into the page-row unit (TILE_HEIGHT for TILE, 1 for
// ROW_MAJOR), so one kernel handles both layouts.
//
// The body lives in a template on `HasMeta` so the `if constexpr` below actually DISCARDS (does not
// instantiate) the unused branch — `kernel_main` is not a template, so an `if constexpr` there would
// still instantiate the metadata branch's TensorAccessor and fail to compile the scalar program.
// `HasValid` is a template param so the no-clamp program keeps the original loop.
template <bool HasMeta, bool HasValid>
static void run_writer() {
    // Per-core runtime args (buffers arrive as Buffer* bindings -> addresses).
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);
    const uint32_t page_bundle_indices_addr = get_arg_val<uint32_t>(3);

    // Common runtime args (same for all cores on this chip). Indices 0-7 are structural; index 8 (and
    // 9, scalar path) carry the per-request values resolved below.
    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(3);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(4);
    const uint32_t Wt = get_common_arg_val<uint32_t>(5);
    const uint32_t cache_HtWt = get_common_arg_val<uint32_t>(6);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(3);
    // [4] is has_valid, consumed as the HasValid template param.
    constexpr auto cache_args = TensorAccessorArgs<5>();
    constexpr auto page_bundle_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t paged_ct_base = page_bundle_args.next_compile_time_args_offset();
    constexpr bool has_paged_cache = get_compile_time_arg_val(paged_ct_base) != 0;
    constexpr uint32_t page_table_cb = get_compile_time_arg_val(paged_ct_base + 1);
    constexpr uint32_t page_size_rows = get_compile_time_arg_val(paged_ct_base + 2);
    constexpr uint32_t page_num_layers = get_compile_time_arg_val(paged_ct_base + 3);
    constexpr uint32_t page_layer_idx = get_compile_time_arg_val(paged_ct_base + 4);
    constexpr uint32_t page_bundle_count = get_compile_time_arg_val(paged_ct_base + 5);

    Noc noc;

    // Resolve the per-request values (in page-row units) from whichever source this program was
    // compiled for.
    uint32_t slot_idx;
    uint32_t kv_actual_global_t;
    // Real tokens as page-rows, rounded up to 32 (zero_padded_kv_cache clears the partial block).
    uint32_t valid_global_t = 0;
    constexpr uint32_t kClampGranularityTokens = 32;
    if constexpr (HasMeta) {
        // Metadata path: NoC-read element [0] (page 0, 4 bytes) of each 1-element uint32 tensor into
        // the L1-scratch CB. Each read targets dst offset 0 (DRAM-read dst-alignment: a 4-byte read into
        // a non-16B-aligned dst offset lands wrong), so we read slot, barrier+extract, then overwrite the
        // same slot with kv_actual_global. Single reserve_back/push_back (a second reserve_back on a
        // single-page CB with no intervening pop would deadlock).
        constexpr uint32_t cb_id_meta = get_compile_time_arg_val(2);
        constexpr uint32_t kMetadataReadBytes = 4;
        // ONE metadata accessor follows the cache accessor in the compile args; it serves both 1-element
        // tensors (identical layout). Gate the offset on HasMeta so this TensorAccessorArgs<> is a
        // *dependent* template-id: `if constexpr` only skips instantiation of the discarded branch's
        // template-parameter-dependent constructs, so the scalar program (no metadata accessor) must
        // not name a fixed out-of-range offset here.
        constexpr uint32_t kMetaArgsOffset = HasMeta ? paged_ct_base + 6 : 0;
        constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
        const uint32_t slot_idx_addr = get_common_arg_val<uint32_t>(8);
        const uint32_t kv_actual_global_addr = get_common_arg_val<uint32_t>(9);
        CircularBuffer cb_meta(cb_id_meta);
        cb_meta.reserve_back(1);

        const auto s_slot = TensorAccessor(meta_args, slot_idx_addr);
        noc.async_read(s_slot, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // Metadata tensor is at a FIXED DRAM address reused every chunk; the RISC data cache may hold the
        // prior chunk's value for this L1 line (barrier orders the DMA, volatile still reads cache). Force
        // a refetch of the freshly-DMA'd value, else an intermittently stale read corrupts the write.
        invalidate_l1_cache();
        slot_idx = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];

        const auto s_kv = TensorAccessor(meta_args, kv_actual_global_addr);
        noc.async_read(s_kv, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();  // same fresh-metadata refetch as above
        kv_actual_global_t = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0] / tile_height;

        if constexpr (HasValid) {
            // Same scratch slot, same fresh-metadata refetch as above.
            const auto s_valid = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(10));
            noc.async_read(s_valid, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            invalidate_l1_cache();
            const uint32_t valid_tokens = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
            valid_global_t = kClampGranularityTokens *
                             ((valid_tokens + kClampGranularityTokens - 1) / kClampGranularityTokens) / tile_height;
        }
        cb_meta.push_back(1);
    } else {
        // Scalar path: per-call values arrive as common runtime args (patched on cache hits).
        slot_idx = get_common_arg_val<uint32_t>(8);
        kv_actual_global_t = get_common_arg_val<uint32_t>(9) / tile_height;
        if constexpr (HasValid) {
            const uint32_t valid_tokens = get_common_arg_val<uint32_t>(10);
            valid_global_t = kClampGranularityTokens *
                             ((valid_tokens + kClampGranularityTokens - 1) / kClampGranularityTokens) / tile_height;
        }
    }

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

    // Real rows are a prefix on every chip, so their end is the staircase above at valid_global_t.
    uint32_t rows_to_write = input_Ht;
    if constexpr (HasValid) {
        const uint32_t end_slab_idx = valid_global_t / chunk_global_t;
        const uint32_t end_chip = (valid_global_t / chunk_local_t) % sp_factor;
        const uint32_t end_offset_t = valid_global_t % chunk_local_t;
        const uint32_t end_idxt =
            end_slab_idx * chunk_local_t +
            (my_sp_coord < end_chip ? chunk_local_t : (my_sp_coord == end_chip ? end_offset_t : 0));
        rows_to_write = (end_idxt > update_idxt) ? (end_idxt - update_idxt) : 0;
        if (rows_to_write > input_Ht) {
            rows_to_write = input_Ht;
        }
    }

    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    CircularBuffer cb(cb_id_out);

    constexpr uint32_t onepage = 1;
    // `cache_args` and `noc` are declared above (shared with the optional metadata read).
    const auto s = TensorAccessor(cache_args, dst_addr);

    if constexpr (has_paged_cache) {
        CircularBuffer table_cb(page_table_cb);
        const uint32_t page_table_l1 = table_cb.get_write_ptr();
        const uint32_t table_bytes = page_bundle_count * sizeof(uint16_t);
        const auto table_reader = TensorAccessor(page_bundle_args, page_bundle_indices_addr);
        noc.async_read(table_reader, CoreLocalMem<uint16_t>(page_table_l1), table_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        invalidate_l1_cache();

        const PagedKVAccessor<decltype(s)> paged_cache{
            s, page_table_l1, page_size_rows, page_num_layers, 1, page_layer_idx};
        const uint32_t num_blocks = num_pages / Wt;
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t block = core_blocks_written + blk;
            const uint32_t row = block % input_Ht;
            const bool keep = !HasValid || row < rows_to_write;
            const uint32_t logical_row = update_idxt + row;
            const auto cursor = paged_cache.cursor(logical_row);
            for (uint32_t w = 0; w < Wt; ++w) {
                cb.wait_front(onepage);
                if (keep) {
                    const uint64_t dst_noc_addr = paged_cache.get_shard_row_noc_addr(
                        cursor, (cursor.row_in_bundle * Wt + w) * page_bytes);
                    noc_async_write(cb.get_read_ptr(), dst_noc_addr, page_bytes);
                    noc_async_writes_flushed();
                }
                cb.pop_front(onepage);
            }
        }
    } else {
        // One block is one page-row (Wt pages) of one head, head-major / row-minor as the reader streams.
        const uint32_t num_blocks = num_pages / Wt;
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t block = core_blocks_written + blk;
            const uint32_t row = block % input_Ht;
            const uint32_t page0 = start_idx + (block / input_Ht) * cache_HtWt + row * Wt;
            const bool keep = !HasValid || row < rows_to_write;
            for (uint32_t w = 0; w < Wt; ++w) {
                cb.wait_front(onepage);
                if (keep) {
                    noc.async_write(cb, s, page_bytes, {}, {.page_id = page0 + w});
                    noc.async_writes_flushed();
                }
                cb.pop_front(onepage);
            }
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(1);
    constexpr bool has_valid = get_compile_time_arg_val(4);
    if constexpr (has_valid) {
        run_writer<has_metadata, true>();
    } else {
        run_writer<has_metadata, false>();
    }
}
