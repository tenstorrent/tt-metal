// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

// Helpers shared verbatim by the H2D (socket_services.cpp) and D2D
// (d2d_stream_service.cpp) stream-service implementations. Both translation
// units live in ttnn/core/tensor and open `namespace tt::tt_metal`, so they pull
// these in with `using namespace stream_service_common;` inside their local
// (anonymous / CMAKE_UNIQUE_NAMESPACE) namespace. Keeping a single copy avoids
// the two services drifting on the socket wire-format (chunk plan) or on which
// dtypes they accept.
namespace tt::tt_metal::stream_service_common {

// Build a single-shard host tensor with zero-initialised data of size `spec`.
// Used purely to feed the mapper at construction time so we can extract a
// TensorTopology + per-shard spec before any user data exists. The bytes are
// never read.
// TODO: replace with a direct "topology from MeshMapperConfig + global shape"
// helper once one exists upstream, so we can skip allocating `spec`-many bytes
// just to throw them away.
inline ttnn::Tensor make_zero_host_tensor(const tt::tt_metal::TensorSpec& spec) {
    const size_t bytes = spec.compute_packed_buffer_size_bytes();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return ttnn::Tensor::from_vector<bfloat16>(std::vector<bfloat16>(bytes / sizeof(bfloat16)), spec);
        case DataType::FLOAT32:
            return ttnn::Tensor::from_vector<float>(std::vector<float>(bytes / sizeof(float)), spec);
        case DataType::INT32:
            return ttnn::Tensor::from_vector<int32_t>(std::vector<int32_t>(bytes / sizeof(int32_t)), spec);
        case DataType::UINT8:
            return ttnn::Tensor::from_vector<uint8_t>(std::vector<uint8_t>(bytes / sizeof(uint8_t)), spec);
        case DataType::UINT16:
            return ttnn::Tensor::from_vector<uint16_t>(std::vector<uint16_t>(bytes / sizeof(uint16_t)), spec);
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
            // Block-float formats pack a shared exponent per group of datums, so the
            // packed byte count is NOT element_count * sizeof. from_vector requires a
            // buffer of exactly logical-volume elements and (per its contract) `float`
            // for block formats; it tilizes + quantizes internally.
            return ttnn::Tensor::from_vector<float>(std::vector<float>(spec.logical_shape().volume()), spec);
        case DataType::UINT32:
            return ttnn::Tensor::from_vector<uint32_t>(std::vector<uint32_t>(bytes / sizeof(uint32_t)), spec);
        // FP8_E4M3 ingestion is not supported on the streaming path (mirrors the
        // TensorToMesh / aggregate_tensor restriction); reject it explicitly.
        case DataType::FP8_E4M3: TT_THROW("StreamService: FP8_E4M3 global_spec data type is not supported");
        case DataType::INVALID: TT_THROW("StreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Picks the largest `pages_per_chunk` (and therefore largest `socket_page_size`)
// that fits in `scratch_cb_size_bytes` and divides `tensor_num_pages` evenly
// (no ragged last chunk), falling back to 1 in the worst case (e.g. a prime
// page count). The result is a socket wire-format contract: sender and receiver
// MUST derive the same plan or the fabric transfer desyncs.
struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
};

inline ChunkPlan derive_chunk_plan(
    uint32_t tensor_page_size, uint32_t tensor_num_pages, uint32_t scratch_cb_size_bytes) {
    TT_FATAL(tensor_page_size > 0, "StreamService: tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "StreamService: backing tensor must have at least one page");
    TT_FATAL(
        scratch_cb_size_bytes >= tensor_page_size,
        "StreamService: scratch_cb_size_bytes ({} B) must be >= tensor page size ({} B)",
        scratch_cb_size_bytes,
        tensor_page_size);

    const uint32_t max_pages_per_chunk_by_cb = scratch_cb_size_bytes / tensor_page_size;
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk_by_cb);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    TT_FATAL(
        pages_per_chunk > 0,
        "StreamService: derived pages_per_chunk == 0 (tensor_page_size={}, tensor_num_pages={}, "
        "scratch_cb_size_bytes={}); the socket FIFO must hold at least one tensor page",
        tensor_page_size,
        tensor_num_pages,
        scratch_cb_size_bytes);
    return ChunkPlan{
        .socket_page_size = pages_per_chunk * tensor_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
    };
}

// Number of cores in an inclusive CoreRange (worker-grid mcast destination count
// + sync-arithmetic target).
inline uint32_t core_range_size(const CoreRange& range) {
    return (range.end_coord.x - range.start_coord.x + 1) * (range.end_coord.y - range.start_coord.y + 1);
}

// Per-coord worker-sync CT-arg block. `data_ready_sem_addr` is the mesh-wide worker-grid
// GlobalSemaphore; `counter_addr` is the per-coord service-core L1 word.
// All zero when disabled.
struct WorkerSyncArgs {
    bool enabled = false;
    uint32_t data_ready_sem_addr = 0;
    uint32_t counter_addr = 0;
    uint32_t mcast_noc_x_start = 0;
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;
};

inline WorkerSyncArgs make_worker_sync_args(
    IDevice* device,
    const CoreRange& worker_cores,
    uint32_t num_workers,
    uint32_t data_ready_sem_addr,
    uint32_t counter_addr,
    bool enabled) {
    WorkerSyncArgs ws;
    const auto start_phys = device->worker_core_from_logical_core(worker_cores.start_coord);
    const auto end_phys = device->worker_core_from_logical_core(worker_cores.end_coord);
    ws.enabled = enabled;
    ws.data_ready_sem_addr = data_ready_sem_addr;
    ws.counter_addr = counter_addr;
    ws.mcast_noc_x_start = static_cast<uint32_t>(start_phys.x);
    ws.mcast_noc_y_start = static_cast<uint32_t>(start_phys.y);
    ws.mcast_noc_x_end = static_cast<uint32_t>(end_phys.x);
    ws.mcast_noc_y_end = static_cast<uint32_t>(end_phys.y);
    ws.num_workers = num_workers;
    return ws;
}

// Claim one service core per participating coord on `mesh`, skipping any cores
// already claimed via ServiceCoreManager. This lets multiple services (e.g. an
// H2D + a D2D sender, or inbound + outbound D2D) share a device without
// re-picking an already-taken core and TT_FATALing in claim(). `side` is a
// human-readable label used in the error message only.
inline std::map<distributed::MeshCoordinate, CoreCoord> claim_service_cores(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::vector<distributed::MeshCoordinate>& coords,
    const char* side) {
    auto& svc = internal::service_core_manager();
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    for (const auto& coord : coords) {
        auto* d = mesh->get_device(coord);
        const auto claimable = svc.get_claimable_cores(d);
        const auto already_claimed = svc.claimed_cores(d->id());
        std::optional<CoreCoord> chosen;
        for (const auto& c : claimable) {
            if (!already_claimed.contains(c)) {
                chosen = c;
                break;
            }
        }
        TT_FATAL(
            chosen.has_value(),
            "StreamService: no unclaimed {} service core on device at coord {} ({} claimable, {} already claimed)",
            side,
            coord,
            claimable.size(),
            already_claimed.size());
        svc.claim(d, {*chosen});
        service_cores.emplace(coord, *chosen);
    }
    return service_cores;
}

}  // namespace tt::tt_metal::stream_service_common
