// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Compile time args and the tile send of writer_local_topk.cpp and writer_local_topk_tree_merge.cpp.
namespace topk_local_writer {

constexpr std::uint32_t receiver_sem_id = get_compile_time_arg_val(0);           // Final core readiness signal
constexpr std::uint32_t sender_sem_id = get_compile_time_arg_val(1);             // Local core completion signal
constexpr std::uint32_t Ht = get_compile_time_arg_val(4);                        // Height tiles to process
constexpr std::uint32_t Kt = get_compile_time_arg_val(6);                        // TopK in tile units (ceil(K/32))
constexpr std::uint32_t values_dfb_index = get_compile_time_arg_val(7);          // Local TopK values output
constexpr std::uint32_t output_ind_dfb_index = get_compile_time_arg_val(8);      // Local TopK indices output
constexpr std::uint32_t final_values_dfb_index = get_compile_time_arg_val(9);    // Final aggregation values buffer
constexpr std::uint32_t final_indices_dfb_index = get_compile_time_arg_val(10);  // Final aggregation indices buffer

constexpr std::uint32_t onetile = 1;

// This core's Kt tiles in the final core's gather CB, which sits at the same L1 address on every core.
FORCE_INLINE std::uint32_t final_slot_addr(
    std::uint32_t final_dfb_index, std::uint32_t slot, std::uint32_t tile_bytes) {
    const DataflowBuffer final_dfb(final_dfb_index);
    return final_dfb.get_write_ptr() + slot * tile_bytes * Kt;
}

// Drains each write before popping its slot: pop_front only moves the read pointer, so the compute's next
// pack could otherwise overwrite the slot while the NoC still reads it.
FORCE_INLINE void send_tiles(
    const Noc& noc,
    DataflowBuffer& src_dfb,
    std::uint32_t num_tiles,
    std::uint32_t tile_bytes,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t dst_base) {
    const UnicastEndpoint remote;
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        src_dfb.wait_front(onetile);
        noc.async_write(
            src_dfb,
            remote,
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = noc_x, .noc_y = noc_y, .addr = dst_base + (i * tile_bytes)});
        noc.async_write_barrier();
        src_dfb.pop_front(onetile);
    }
}

}  // namespace topk_local_writer
