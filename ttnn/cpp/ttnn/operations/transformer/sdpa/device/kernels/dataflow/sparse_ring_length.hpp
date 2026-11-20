// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Runtime logical length of the sparse ring op (logical_n_tensor, SPARSE_LOGICAL_N_TENSOR): the reader and the writer
// read the [1, 1, 1, 1] uint32 tensor at every dispatch (the host rewrites it in place between calls and trace
// replays) and derive what the host derives from logical_n without it.

#include <cstdint>

#include "metadata_scalar_read.hpp"
#include "ring_joint_kv_pad_derivation.hpp"

namespace sparse_ring_length {

struct Length {
    uint32_t logical_n;
    uint32_t logical_nt;
    uint32_t active_ring_iter_mask;
};

// `scratch`: an L1 address at a CB page base the caller is not using yet. The length is clamped to [1, padded length]
// (the host validation's range). The ring mask is the host's build_ring_work_masks
// (sparse_ring_sdpa_program_factory.cpp) walked on device: an iteration is active when one of its K pieces starts
// inside logical_n.
template <
    bool FullMesh,
    uint32_t ring_size,
    uint32_t kv_local_padded_Nt,
    uint32_t Sk_chunk_t,
    uint32_t num_local_k_chunks,
    uint32_t chunk_size_t,
    uint32_t kv_region_Nt,
    typename NocT,
    typename AccessorArgsT>
inline Length read(
    NocT& noc,
    const AccessorArgsT& accessor_args,
    uint32_t tensor_addr,
    uint32_t scratch,
    const RingIdSequencer& seq,
    uint32_t mesh_rows,
    uint32_t mesh_cols,
    ttnn::ccl::snake_ring::Orientation snake_orientation) {
    constexpr uint32_t padded_n = kv_local_padded_Nt * 32 * ring_size;
    const uint32_t live = trace_metadata::read_metadata_scalar_u32(noc, accessor_args, tensor_addr, scratch);
    ASSERT(live >= 1 && live <= padded_n);
    Length len;
    len.logical_n = live < 1 ? 1 : (live > padded_n ? padded_n : live);
    len.logical_nt = (len.logical_n + 31) / 32;
    const auto masks = ttnn::operations::transformer::sdpa::ring_joint::build_ring_work_masks_device<FullMesh>({
        .transport_rank = seq.ring_index,
        .ring_size = ring_size,
        .mesh_rows = mesh_rows,
        .mesh_cols = mesh_cols,
        .snake_orientation = snake_orientation,
        .backward_writes_expected = seq.expected[0],
        .forward_writes_expected = seq.expected[1],
        .num_local_k_chunks = num_local_k_chunks,
        .k_chunk_tile_count = Sk_chunk_t,
        .kv_local_padded_Nt = kv_local_padded_Nt,
        .kernel_chunked = false,
        .q_chunk_group_tile_count = chunk_size_t,
        .kv_region_Nt = kv_region_Nt,
        .logical_nt = len.logical_nt,
        .num_joint_k_chunks = 0,
        .joint_seq_len = 0,
        .kv_pad_rotation_enabled = false,
        .kernel_is_causal = false,
        .is_balanced = false,
    });
    len.active_ring_iter_mask = masks.active_ring_iter_mask;
    return len;
}

}  // namespace sparse_ring_length
