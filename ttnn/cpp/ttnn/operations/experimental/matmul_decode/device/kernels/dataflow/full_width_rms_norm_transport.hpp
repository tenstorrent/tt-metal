// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Stats transport for the fused RMSNorm epilogue, run by the writer RISC on every producer core.
//
// Runtime metadata at metadata_arg_base:
//   [local_group_count, hub_group_count, hub_contributor_count]
//   local group: [local_tile_offset, tile_count, hub_x, hub_y, hub_gathered_slot, hub_row_stride]
//   hub group:   [contributor_count, gathered_slot_base,
//                 contributor_x, contributor_y, contributor_scale_slot, contributor_row_stride, ...]
//
// CB ownership remains single-producer / single-consumer. Compute publishes local group statistics;
// writers move them into group-hub gather slots; hub compute publishes scales; writers return each
// scale only to that group's contributors. The local fragment fields are consumed by compute in Task 3.
inline void run_full_width_rms_norm_transport(uint32_t metadata_arg_base) {
    constexpr uint32_t cb_rms_local = get_named_compile_time_arg_val("cb_rms_local");
    constexpr uint32_t cb_rms_gathered = get_named_compile_time_arg_val("cb_rms_gathered");
    constexpr uint32_t cb_rms_scale_src = get_named_compile_time_arg_val("cb_rms_scale_src");
    constexpr uint32_t cb_rms_scale = get_named_compile_time_arg_val("cb_rms_scale");
    constexpr uint32_t cb_rms_reduce_scaler = get_named_compile_time_arg_val("cb_rms_reduce_scaler");
    constexpr uint32_t arrival_sem_id = get_named_compile_time_arg_val("rms_arrival_sem");
    constexpr uint32_t scale_ready_sem_id = get_named_compile_time_arg_val("rms_scale_ready_sem");
    constexpr uint32_t M_tiles = get_named_compile_time_arg_val("rms_m_tiles");
    constexpr uint32_t local_tile_size = get_named_compile_time_arg_val("rms_local_tile_size");
    constexpr uint32_t reduce_tile_size = get_named_compile_time_arg_val("rms_reduce_tile_size");
    constexpr uint32_t local_group_arg_words = 6;

    Noc noc;
    CircularBuffer rms_local(cb_rms_local);
    CircularBuffer rms_gathered(cb_rms_gathered);
    CircularBuffer rms_scale_src(cb_rms_scale_src);
    CircularBuffer rms_scale(cb_rms_scale);
    Semaphore<> arrival_sem(arrival_sem_id);
    Semaphore<> scale_ready_sem(scale_ready_sem_id);
    UnicastEndpoint endpoint;

    uint32_t arg = metadata_arg_base;
    const uint32_t local_group_count = get_arg_val<uint32_t>(arg++);
    const uint32_t hub_group_count = get_arg_val<uint32_t>(arg++);
    const uint32_t hub_contributor_count = get_arg_val<uint32_t>(arg++);
    const uint32_t local_groups_arg_base = arg;
    const uint32_t hub_groups_arg_base = local_groups_arg_base + local_group_arg_words * local_group_count;

    // Hubs prepare gathered destinations before contributors write. One readiness increment per
    // local group lets a producer wait for all of its (possibly different) hubs.
    const uint32_t local_scale_tiles = M_tiles * local_group_count;
    rms_scale.reserve_back(local_scale_tiles);
    if (hub_group_count != 0) {
        const uint32_t gathered_tiles = M_tiles * hub_contributor_count;
        rms_gathered.reserve_back(gathered_tiles);
        auto* packed_dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rms_gathered.get_write_ptr());
        for (uint32_t byte = 0; byte < gathered_tiles * reduce_tile_size; byte += sizeof(uint32_t)) {
            packed_dst[byte / sizeof(uint32_t)] = 0;
        }

        uint32_t hub_arg = hub_groups_arg_base;
        for (uint32_t hg = 0; hg < hub_group_count; ++hg) {
            const uint32_t contributor_count = get_arg_val<uint32_t>(hub_arg++);
            hub_arg++;  // gathered_slot_base
            for (uint32_t c = 0; c < contributor_count; ++c) {
                const uint32_t contributor_x = get_arg_val<uint32_t>(hub_arg++);
                const uint32_t contributor_y = get_arg_val<uint32_t>(hub_arg++);
                hub_arg += 2;  // contributor_scale_slot, contributor_row_stride
                scale_ready_sem.up(noc, contributor_x, contributor_y, 1);
            }
        }
        noc.async_atomic_barrier();
    }
    scale_ready_sem.wait(local_group_count);
    scale_ready_sem.set(0);

    // Each producer writes its partial statistics into disjoint slots on the corresponding hubs.
    rms_local.wait_front(M_tiles * local_group_count);
    for (uint32_t lg = 0; lg < local_group_count; ++lg) {
        const uint32_t fragment_arg = local_groups_arg_base + lg * local_group_arg_words;
        const uint32_t hub_x = get_arg_val<uint32_t>(fragment_arg + 2);
        const uint32_t hub_y = get_arg_val<uint32_t>(fragment_arg + 3);
        const uint32_t hub_gathered_slot = get_arg_val<uint32_t>(fragment_arg + 4);
        const uint32_t hub_row_stride = get_arg_val<uint32_t>(fragment_arg + 5);
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t src_offset = (mt * local_group_count + lg) * local_tile_size;
            const uint32_t dst_offset = (mt * hub_row_stride + hub_gathered_slot) * reduce_tile_size;
            noc.async_write(
                rms_local,
                endpoint,
                local_tile_size,
                {.offset_bytes = src_offset},
                {.noc_x = hub_x, .noc_y = hub_y, .addr = rms_gathered.get_write_ptr() + dst_offset});
        }
        noc.async_write_barrier();
        arrival_sem.up(noc, hub_x, hub_y, 1);
    }
    noc.async_atomic_barrier();

    if (hub_group_count != 0) {
        arrival_sem.wait(hub_contributor_count);
        arrival_sem.set(0);
        rms_gathered.push_back(M_tiles * hub_contributor_count);

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_rms_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_SCALAR>();

        // Compute produces one scale per owned group. Direct unicasts avoid overlapping
        // whole-grid multicast reservations and target contributor cores only.
        rms_scale_src.wait_front(M_tiles * hub_group_count);
        uint32_t hub_arg = hub_groups_arg_base;
        for (uint32_t hg = 0; hg < hub_group_count; ++hg) {
            const uint32_t contributor_count = get_arg_val<uint32_t>(hub_arg++);
            hub_arg++;  // gathered_slot_base
            for (uint32_t c = 0; c < contributor_count; ++c) {
                const uint32_t contributor_x = get_arg_val<uint32_t>(hub_arg++);
                const uint32_t contributor_y = get_arg_val<uint32_t>(hub_arg++);
                const uint32_t contributor_scale_slot = get_arg_val<uint32_t>(hub_arg++);
                const uint32_t contributor_row_stride = get_arg_val<uint32_t>(hub_arg++);
                for (uint32_t mt = 0; mt < M_tiles; ++mt) {
                    const uint32_t src_offset = (mt * hub_group_count + hg) * local_tile_size;
                    const uint32_t dst_offset =
                        (mt * contributor_row_stride + contributor_scale_slot) * local_tile_size;
                    noc.async_write(
                        use<CircularBuffer::AddrSelector::READ_PTR>(rms_scale_src),
                        endpoint,
                        local_tile_size,
                        {.offset_bytes = src_offset},
                        {.noc_x = contributor_x,
                         .noc_y = contributor_y,
                         .addr = rms_scale.get_write_ptr() + dst_offset});
                }
            }
        }
        noc.async_write_barrier();

        // Signal only after every scale payload is committed at its contributor destination.
        hub_arg = hub_groups_arg_base;
        for (uint32_t hg = 0; hg < hub_group_count; ++hg) {
            const uint32_t contributor_count = get_arg_val<uint32_t>(hub_arg++);
            hub_arg++;  // gathered_slot_base
            for (uint32_t c = 0; c < contributor_count; ++c) {
                const uint32_t contributor_x = get_arg_val<uint32_t>(hub_arg++);
                const uint32_t contributor_y = get_arg_val<uint32_t>(hub_arg++);
                hub_arg += 2;
                scale_ready_sem.up(noc, contributor_x, contributor_y, 1);
            }
        }
        noc.async_atomic_barrier();
        rms_scale_src.pop_front(M_tiles * hub_group_count);
    }

    scale_ready_sem.wait(local_group_count);
    scale_ready_sem.set(0);
    rms_scale.push_back(local_scale_tiles);
    rms_local.pop_front(M_tiles * local_group_count);
}
