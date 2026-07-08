// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul-signal aggregation kernel for the writer-signals-matmul path (Option W).
//
// One aggregator runs per all-gather direction on the receiving device. Each of the N AG writer
// workers in that direction fabric-increments its own per-worker semaphore on this core as soon as
// its portion of a k-block lands (ordered after the data on that worker's fabric connection). This
// aggregator waits for all N per-worker semaphores to advance, which means the whole k-block has
// landed, then increments the matmul cores' single direction semaphore exactly once - decoupling the
// matmul from the AG reader's forwarding pace while keeping the matmul cores at just 3 semaphores.
//
// The iteration cadence mirrors the reader's remote-receive loop so that the number and order of
// matmul signals matches exactly what the (reader-signaled) legacy path produced.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using ttnn::ccl::Topology;

// Each AG writer worker fires this many per-worker incs per chunk (one per row-band). The aggregator
// waits for all of them (event_target advances by IN0_SUB_CHUNKS per chunk) before signaling the
// matmul once per chunk. Default 1 = single inc per chunk (legacy). Must match the writer.
#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(1);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(2);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(3));
constexpr bool direction = get_compile_time_arg_val(4);  // 1 is forward, 0 is backward
constexpr uint32_t num_ag_workers = get_compile_time_arg_val(5);
constexpr uint32_t num_mm_cores = get_compile_time_arg_val(6);

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_batches = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_cores_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_block_ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_signal_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t device_k_block_counts[ring_size];
    for (uint32_t d = 0; d < ring_size; d++) {
        device_k_block_counts[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Per-worker semaphore addresses (this core's L1) that the AG writer workers increment.
    // GlobalSemaphore L1 addresses (already resolved host-side); do NOT get_semaphore() them.
    volatile tt_l1_ptr uint32_t* per_worker_sem_ptrs[num_ag_workers];
    for (uint32_t w = 0; w < num_ag_workers; w++) {
        per_worker_sem_ptrs[w] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    }

    // Matmul receiver core NOC coords (this device) to signal, [x0, y0, x1, y1, ...].
    uint32_t mm_core_noc_coords[num_mm_cores * 2];
    for (uint32_t i = 0; i < num_mm_cores * 2; i++) {
        mm_core_noc_coords[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Number of remote slices this direction receives (mirrors the reader).
    uint32_t slices_expected = 0;
    if (topology == Topology::Linear) {
        slices_expected = direction == 1 ? num_targets_forward_direction : num_targets_backward_direction;
    } else {  // Ring
        slices_expected = direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;
    }

    // Split-forwarding: on an even ring the diametric slice arrives split across both links, so the
    // upstream writer fabric-increments this direction's per-worker sems only for its half. Mirror
    // the reader's receive accounting exactly so event_target tracks what the writer actually sends.
    bool split_forwarding_enabled = (topology == Topology::Ring) && (ring_size % 2 == 0) && (ring_size > 2);
    if (split_forwarding_enabled && direction == 1) {
        slices_expected++;
    }

    uint32_t padded_M_tiles = round_up(input_tensor_Ht, mm_cores_y);
    uint32_t M_tiles_per_core = padded_M_tiles / mm_cores_y;
    uint32_t M_blocks_per_core = div_up(M_tiles_per_core, mm_block_ht);

    uint32_t event_target = 0;
    for (uint32_t b_idx = 0; b_idx < num_batches; b_idx++) {
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            uint32_t slices_received = 0;
            while (slices_received < slices_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slices_received, ring_size);
                uint32_t num_chunks = device_k_block_counts[actual_sender_chip_id];
                // The diametric slice (last one received) lands split across both links; wait on only
                // this direction's half (direction 0 = first half, direction 1 = second half).
                bool is_split_received_slice =
                    split_forwarding_enabled && (slices_received == slices_expected - 1) && (num_chunks >= 2);
                uint32_t first_half_chunks = num_chunks / 2;
                for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    bool receive_this_chunk =
                        !is_split_received_slice ||
                        (direction == 0 ? (chunk_idx < first_half_chunks) : (chunk_idx >= first_half_chunks));
                    if (!receive_this_chunk) {
                        continue;
                    }
                    // Writer fires IN0_SUB_CHUNKS per-worker incs per chunk (one per row-band). Signal
                    // the matmul once per band as it lands, so the injector can read band s while band
                    // s+1 is still on the fabric.
                    for (uint32_t sub = 0; sub < IN0_SUB_CHUNKS; sub++) {
                        event_target++;
                        // Wait for all N workers' portion of this band to land.
                        {
                            DeviceZoneScopedN("AGG-WAIT-N");
                            for (uint32_t w = 0; w < num_ag_workers; w++) {
                                noc_semaphore_wait_min(per_worker_sem_ptrs[w], event_target);
                            }
                        }
                        for (uint32_t c = 0; c < num_mm_cores; c++) {
                            uint64_t mm_sem_noc_addr = get_noc_addr(
                                mm_core_noc_coords[c * 2], mm_core_noc_coords[c * 2 + 1], mm_signal_sem_addr);
                            noc_semaphore_inc(mm_sem_noc_addr, 1);
                        }
                    }
                }
                slices_received++;
            }
        }
    }
    // event_target counts up monotonically and each wait gates on it, so the per-worker sems must
    // start every invocation at 0. Zero them here (all increments for this run have been observed by
    // the final wait_min) so a trace replay begins clean instead of inheriting the prior run's value.
    for (uint32_t w = 0; w < num_ag_workers; w++) {
        noc_semaphore_set(per_worker_sem_ptrs[w], 0);
    }
}
