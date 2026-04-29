// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for streaming_helpers_dataflow.hpp
// Do not include directly - include streaming_helpers_dataflow.hpp instead

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"

namespace dataflow_kernel_lib {

template <uint32_t cb_in, ReduceDim reduce_dim, typename Accessor>
FORCE_INLINE void stream_axis_blocks(
    const Accessor& accessor,
    uint32_t preserved_start,
    uint32_t preserved_end,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t BLOCK_SIZE) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_ROW || reduce_dim == ReduceDim::REDUCE_COL,
        "stream_axis_blocks supports only REDUCE_ROW (collapse W) and REDUCE_COL (collapse H). "
        "REDUCE_SCALAR has no preserved axis to slice.");

    constexpr bool collapse_w = (reduce_dim == ReduceDim::REDUCE_ROW);
    const uint32_t reduce_extent = collapse_w ? Wt : Ht;

    ASSERT(BLOCK_SIZE > 0);
    ASSERT(reduce_extent % BLOCK_SIZE == 0);
    const uint32_t num_blocks = reduce_extent / BLOCK_SIZE;

    for (uint32_t outer = preserved_start; outer < preserved_end; ++outer) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            for (uint32_t inner = 0; inner < BLOCK_SIZE; ++inner) {
                uint32_t tile_id;
                if constexpr (collapse_w) {
                    tile_id = outer * Wt + b * BLOCK_SIZE + inner;
                } else {
                    tile_id = (b * BLOCK_SIZE + inner) * Wt + outer;
                }
                cb_reserve_back(cb_in, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_in);
                noc_async_read_tile(tile_id, accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_in, 1);
            }
        }
    }
}

}  // namespace dataflow_kernel_lib
