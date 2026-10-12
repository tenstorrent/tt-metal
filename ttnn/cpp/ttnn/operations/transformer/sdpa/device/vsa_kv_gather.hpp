// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// vsa_ring_sdpa's K/V ring all-gather (kernels vsa_kv_gather_reader/writer.cpp, forked from all_gather_async's
// multi-worker kernels): two [1, H, T_local, d] inputs into two [1, H, T_local * ring_size, d] persistent outputs,
// Ring topology, `num_workers_per_link` workers per direction per link behind a fabric MUX (one worker per link:
// direct fabric connections, no MUX). Worker g (global id link * workers + worker, the same for both directions)
// owns the tile rows [row_ranges[g]) of every slice and forwards them TOKEN-MAJOR (per tile row: K of every head,
// then V of every head), bumping the receiver's out_ready_sem every `chunks_per_sync` packets of `tiles_per_packet`
// tiles. The fused-op signaler gets one signal per landed slice (the OpSignaler protocol RingSDPAOpReceiver reads).
// Cores are placed from (0, 0) in row-major order: link, direction, MUX core, then the workers.
struct VsaKvGatherArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> all_cores;  // logical, in placement order
    uint32_t num_links = 0;
    uint32_t num_directions_per_link = 2;
    uint32_t num_workers_per_direction = 0;
    uint32_t num_mux_cores_per_direction_per_link = 0;
    uint32_t num_cores_per_link = 0;
    uint32_t tiles_per_packet = 1;
    uint32_t chunks_per_sync = 1;
    std::vector<std::pair<uint32_t, uint32_t>> row_ranges;  // [first, end) tile rows per global worker id

    // logical core of (link, direction, worker)
    tt::tt_metal::CoreCoord worker_core(uint32_t link, uint32_t dir, uint32_t worker) const {
        return all_cores.at(
            link * num_cores_per_link + dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction) +
            num_mux_cores_per_direction_per_link + worker);
    }
};

VsaKvGatherArtifacts build_vsa_kv_gather(
    tt::tt_metal::Program& program,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gathered_k,
    const Tensor& gathered_v,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    const std::vector<GlobalSemaphore>& semaphore,  // [direction 0, direction 1] out_ready semaphores
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::experimental::ccl::AllGatherFusedOpSignaler& fused_op_signaler,
    uint32_t num_workers_per_link,
    std::optional<uint32_t> chunks_per_sync);

// Cache hit: re-apply the buffer and semaphore addresses (excluded from the program hash).
void vsa_kv_gather_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaKvGatherArtifacts& artifacts,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gathered_k,
    const Tensor& gathered_v);

}  // namespace ttnn::prim
