// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/assert.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"

namespace trace_metadata {

// Read one per-chunk metadata scalar -- page 0 of a 1-element uint32 DRAM tensor -- into L1 scratch and
// return its value.
//
// Such a tensor sits at a FIXED DRAM address that the host refreshes in place between trace replays,
// which is what lets a single captured program pick up new per-chunk scalars instead of having them baked
// in as runtime args. Two steps of the sequence are easy to omit and both fail *silently*, with
// plausible-but-wrong data rather than an error:
//
//   - `async_read_barrier()` orders the DMA but does NOT invalidate the RISC data cache. Since the
//     address is reused every chunk, a cached L1 line hands back the PRIOR chunk's value. Hence
//     `invalidate_l1_cache()` between the barrier and the load.
//   - the load itself must be volatile, or it can be hoisted above the DMA.
//
// `dst_l1_addr` is deliberately the caller's choice: which CB's L1 may be borrowed as scratch, and at
// which offset, is kernel-specific -- these small reads only land correctly at a CB page base on some
// platforms -- so the caller passes an address it has reasoned about rather than this helper guessing.
template <typename NocT, typename AccessorArgsT>
inline uint32_t read_metadata_scalar_u32(
    NocT& noc, const AccessorArgsT& accessor_args, uint32_t tensor_addr, uint32_t dst_l1_addr) {
    const auto accessor = TensorAccessor(accessor_args, tensor_addr);
    noc.async_read(accessor, CoreLocalMem<uint8_t>(dst_l1_addr), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    return CoreLocalMem<volatile uint32_t>(dst_l1_addr)[0];
}

inline uint32_t bounded_cache_batch_idx(
    uint32_t slot_id, uint32_t num_layers, uint32_t layer_idx, uint32_t cache_batch_extent) {
    const bool static_args_valid = num_layers > 0 && layer_idx < num_layers;
    const bool layer_fits_cache = layer_idx < cache_batch_extent;
    const bool slot_fits_cache =
        static_args_valid && layer_fits_cache && slot_id <= (cache_batch_extent - 1 - layer_idx) / num_layers;
    ASSERT(slot_fits_cache);
    // Device assertions are optional; keep the resulting NoC address inside the allocation when disabled.
    return slot_fits_cache ? slot_id * num_layers + layer_idx : 0;
}

inline uint32_t bounded_kv_actual_isl(
    uint32_t kv_actual_isl, uint32_t chunk_global_tile_rows, uint32_t cache_global_tile_rows) {
    constexpr uint32_t tile_height = 32;
    const bool tile_aligned = kv_actual_isl % tile_height == 0;
    const uint32_t kv_actual_tile_rows = kv_actual_isl / tile_height;
    const bool geometry_valid = chunk_global_tile_rows > 0 && chunk_global_tile_rows <= cache_global_tile_rows;
    // The metadata identifies the existing KV prefix, not the number of valid rows in the current Q
    // chunk. A partially filled final chunk may start less than one padded chunk from the cache end;
    // validate the start here and clamp the derived logical extent below.
    const bool metadata_valid = geometry_valid && tile_aligned && kv_actual_tile_rows < cache_global_tile_rows;
    ASSERT(metadata_valid);
    // Treat invalid metadata as the first chunk when device assertions are disabled. Every metadata
    // consumer uses this fallback, keeping CCL producer/consumer counts and SDPA work masks consistent.
    return metadata_valid ? kv_actual_isl : 0;
}

// Metadata does not carry the current chunk's valid-row count. Its padded extent is authoritative except
// at the physical cache boundary, where the remaining cache rows describe a partially filled final chunk.
inline uint32_t logical_tile_rows_clamped_to_cache(
    uint32_t kv_actual_isl, uint32_t chunk_global_tile_rows, uint32_t cache_global_tile_rows) {
    constexpr uint32_t tile_height = 32;
    const uint32_t kv_actual_tile_rows = kv_actual_isl / tile_height;
    const uint32_t remaining_tile_rows = cache_global_tile_rows - kv_actual_tile_rows;
    return kv_actual_tile_rows +
           (chunk_global_tile_rows < remaining_tile_rows ? chunk_global_tile_rows : remaining_tile_rows);
}

}  // namespace trace_metadata
