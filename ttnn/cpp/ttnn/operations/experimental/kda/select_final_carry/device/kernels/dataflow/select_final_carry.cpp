// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Copy this worker's tiles of the state after the chronologically last valid token: the owning rank's gathered
// final when the valid interval ends in a separated tail, otherwise the completed distributed prefix.
template <
    uint32_t state_tiles,
    uint32_t tiles_per_core,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void dataflow(uint32_t first_tile) {
    const auto rank_finals = TensorAccessor(tensor::rank_finals);
    const auto prefix_final = TensorAccessor(tensor::prefix_final);
    const auto output = TensorAccessor(tensor::output);
    const auto actual_start = TensorAccessor(tensor::actual_start);
    DataflowBuffer staging(dfb::staging);
    Noc noc;

    staging.reserve_back(tiles_per_core);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging.get_write_ptr());
    noc.async_read(actual_start, staging, sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
        noc.async_read(end, staging, sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        topology = kda_chronology::derive_interval(start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
    }

    const uint32_t end_tile = first_tile + tiles_per_core < state_tiles ? first_tile + tiles_per_core : state_tiles;
    const uint32_t tile_bytes = staging.get_entry_size();
    for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        const uint32_t offset = (tile - first_tile) * tile_bytes;
        if (topology.split) {
            noc.async_read(
                rank_finals,
                staging,
                tile_bytes,
                {.page_id = topology.final_owner * state_tiles + tile},
                {.offset_bytes = offset});
        } else {
            noc.async_read(prefix_final, staging, tile_bytes, {.page_id = tile}, {.offset_bytes = offset});
        }
    }
    noc.async_read_barrier();
    for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        noc.async_write(
            staging, output, tile_bytes, {.offset_bytes = (tile - first_tile) * tile_bytes}, {.page_id = tile});
    }
    noc.async_write_barrier();
    staging.push_back(tiles_per_core);
    staging.wait_front(tiles_per_core);
    staging.pop_front(tiles_per_core);
}
