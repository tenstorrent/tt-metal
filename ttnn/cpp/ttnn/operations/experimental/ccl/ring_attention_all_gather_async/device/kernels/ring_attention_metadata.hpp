// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared reader for the trace-safe ring-attention metadata tensor. The replicated uint32 DRAM payload is
// [slot_id, actual_start, actual_end]; kernels read the first two elements on-device so captured traces can
// replay across chunks. This header owns the read helper and byte count used by all-gather and ring_joint SDPA.

#pragma once

#include <cstdint>

#include "api/dataflow/noc.h"              // Noc
#include "api/dataflow/circular_buffer.h"  // CircularBuffer
#include "api/core_local_mem.h"            // CoreLocalMem
#include "api/tensor/tensor_accessor.h"    // TensorAccessor

namespace ttnn::ring_attention {

// Bytes NoC-read from the metadata tensor: metadata[0] (slot) + metadata[1] (kv_actual_isl). metadata[2]
// (actual_end) is unused on-device (logical_n stays a host arg).
constexpr uint32_t kMetadataReadBytes = 2 * sizeof(uint32_t);

struct RingMetadata {
    uint32_t slot;       // metadata[0] = cache-user slot
    uint32_t kv_actual;  // metadata[1] = actual_start = kv_actual_isl (tile-aligned)
};

// Read [slot, kv_actual] from the metadata tensor and return them. `meta_args` is the tensor's compile-time
// TensorAccessorArgs; `metadata_addr` its DRAM base (a common runtime arg on the SDPA kernels, a per-core
// runtime arg on the all-gather kernels); `scratch_cb` is a real, currently-unused CB used as L1 landing.
//
// Use a real CB at offset 0 for the NoC landing. On this platform, stack-buffer reads hang, nonzero CB
// offsets read back zero, and a tiny dedicated metadata CB was unreliable. Callers pass an allocated but
// unused CB; the consumer overwrites it before normal use.
template <typename MetaAccessorArgs>
inline RingMetadata read_ring_metadata(
    Noc& noc, const MetaAccessorArgs& meta_args, uint32_t metadata_addr, const CircularBuffer& scratch_cb) {
    const auto s_meta = TensorAccessor(meta_args, metadata_addr);
    noc.async_read(s_meta, scratch_cb, kMetadataReadBytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    CoreLocalMem<volatile uint32_t> meta(scratch_cb.get_write_ptr());
    return {meta[0], meta[1]};
}

// Per-(batch, head) valid page count the fused ring_joint gather may move, from a tile-aligned kv_actual:
// take logical_nt (= kv_actual/TILE + one full chunk of chunk_global_tiles), round up to whole chunk-slabs,
// and scale to local tiles. Shared by the all-gather reader and writer so their gather clamp stays identical.
// (The SDPA work-plan derives the same logical_nt for its attention extent via compute_logical_nt; the two
// live in different op layers and can't share this helper without a layering inversion, so keep them in sync.)
inline uint32_t compute_ring_gather_valid_Ht(
    uint32_t kv_actual, uint32_t chunk_local_tiles, uint32_t ring_size, uint32_t tile_height) {
    const uint32_t chunk_global_tiles = chunk_local_tiles * ring_size;
    const uint32_t logical_nt = kv_actual / tile_height + chunk_global_tiles;
    const uint32_t valid_slabs = (logical_nt + chunk_global_tiles - 1) / chunk_global_tiles;
    return valid_slabs * chunk_local_tiles;
}

}  // namespace ttnn::ring_attention
