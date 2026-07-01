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
// Compile args: [0]=cb_id_out, [1]=has_metadata, [2]=cb_id_meta, [3]=tile_height, [4..]=cache
// accessor, then (metadata path only) ONE metadata accessor (the two 1-element tensors share an
// identical layout, so the same accessor serves both reads). tile_height divides kv tokens into the
// page-row unit (TILE_HEIGHT for TILE, 1 for ROW_MAJOR), so one kernel handles both layouts.
//
// The body lives in a template on `HasMeta` so the `if constexpr` below actually DISCARDS (does not
// instantiate) the unused branch — `kernel_main` is not a template, so an `if constexpr` there would
// still instantiate the metadata branch's TensorAccessor and fail to compile the scalar program.
template <bool HasMeta>
static void run_writer() {
    // Per-core runtime args (buffers arrive as Buffer* bindings -> addresses).
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);

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
    constexpr auto cache_args = TensorAccessorArgs<4>();

    Noc noc;

    // Resolve the two per-request values (slot_idx, and kv_actual_global already divided into the
    // page-row unit) from whichever source this program was compiled for.
    uint32_t slot_idx;
    uint32_t kv_actual_global_t;
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
        constexpr uint32_t kMetaArgsOffset = HasMeta ? cache_args.next_compile_time_args_offset() : 0;
        constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
        const uint32_t slot_idx_addr = get_common_arg_val<uint32_t>(8);
        const uint32_t kv_actual_global_addr = get_common_arg_val<uint32_t>(9);
        CircularBuffer cb_meta(cb_id_meta);
        cb_meta.reserve_back(1);

        const auto s_slot = TensorAccessor(meta_args, slot_idx_addr);
        noc.async_read(s_slot, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        slot_idx = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];

        const auto s_kv = TensorAccessor(meta_args, kv_actual_global_addr);
        noc.async_read(s_kv, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        kv_actual_global_t = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0] / tile_height;
        cb_meta.push_back(1);
    } else {
        // Scalar path: per-call values arrive as common runtime args (patched on cache hits).
        slot_idx = get_common_arg_val<uint32_t>(8);
        kv_actual_global_t = get_common_arg_val<uint32_t>(9) / tile_height;
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
    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
    const uint32_t start_id =
        start_idx + (core_blocks_written / input_Ht) * cache_HtWt + (core_blocks_written % input_Ht) * Wt;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    CircularBuffer cb(cb_id_out);

    constexpr uint32_t onepage = 1;
    // `cache_args` and `noc` are declared above (shared with the optional metadata read).
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

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(1);
    run_writer<has_metadata>();
}
