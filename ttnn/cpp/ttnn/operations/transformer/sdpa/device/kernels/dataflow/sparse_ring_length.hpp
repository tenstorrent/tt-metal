// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Runtime logical length of the sparse ring op (logical_n_tensor, SPARSE_LOGICAL_N_TENSOR): the reader reads the
// [1, 1, 1, 1] uint32 tensor at every dispatch (the host rewrites it in place between calls and trace replays) and
// derives what the host derives from logical_n without it; it hands logical_nt and the ring mask to the writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace sparse_ring_length {

struct Length {
    uint32_t logical_n;
    uint32_t logical_nt;
    uint32_t active_ring_iter_mask;
};

// `tensor_addr`: the tensor's buffer address; page 0 of an interleaved DRAM buffer is in bank 0 (validated on the
// host). `scratch`: an L1 address at a CB page base the caller is not using yet. The length is clamped to [1, padded
// length] (the host validation's range). The ring mask is the host's build_ring_work_masks
// (sparse_ring_sdpa_program_factory.cpp): ring iteration i is active when its shard's first tile lies inside
// logical_n, i.e. logical_nt > start_i, with start_i = (the shard iteration i visits) * kv_local_padded_Nt, which the
// host passes in the common args from `starts_arg` on (it depends on the device's ring position, not on the length).
template <uint32_t ring_size, uint32_t kv_local_padded_Nt>
inline Length read(uint32_t tensor_addr, uint32_t scratch, uint32_t starts_arg) {
    constexpr uint32_t padded_n = kv_local_padded_Nt * 32 * ring_size;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, tensor_addr), scratch, sizeof(uint32_t));
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t live = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];
    Length len;
    len.logical_n = live < 1 ? 1 : (live > padded_n ? padded_n : live);
    len.logical_nt = (len.logical_n + 31) / 32;
    len.active_ring_iter_mask = 0;
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (len.logical_nt > get_common_arg_val<uint32_t>(starts_arg + i)) {
            len.active_ring_iter_mask |= 1u << i;
        }
    }
    return len;
}

}  // namespace sparse_ring_length
