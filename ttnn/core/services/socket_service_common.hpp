// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal helpers shared by h2d_socket_service.cpp and d2h_socket_service.cpp.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

// Zero-filled host tensor of size `spec`, used to feed the mapper at construction.
inline Tensor make_zero_host_tensor(const TensorSpec& spec) {
    const size_t bytes = spec.compute_packed_buffer_size_bytes();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_vector<bfloat16>(std::vector<bfloat16>(bytes / sizeof(bfloat16)), spec);
        case DataType::FLOAT32: return Tensor::from_vector<float>(std::vector<float>(bytes / sizeof(float)), spec);
        case DataType::INT32: return Tensor::from_vector<int32_t>(std::vector<int32_t>(bytes / sizeof(int32_t)), spec);
        case DataType::UINT8: return Tensor::from_vector<uint8_t>(std::vector<uint8_t>(bytes / sizeof(uint8_t)), spec);
        case DataType::UINT16:
            return Tensor::from_vector<uint16_t>(std::vector<uint16_t>(bytes / sizeof(uint16_t)), spec);
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
            // Block-float formats pack a shared exponent per group of datums, so the
            // packed byte count is NOT element_count * sizeof. from_vector requires a
            // buffer of exactly logical-volume elements and (per its contract) `float`
            // for block formats; it tilizes + quantizes internally.
            return Tensor::from_vector<float>(std::vector<float>(spec.logical_shape().volume()), spec);
        case DataType::UINT32:
            return Tensor::from_vector<uint32_t>(std::vector<uint32_t>(bytes / sizeof(uint32_t)), spec);
        case DataType::FP8_E4M3: TT_THROW("StreamService: FP8_E4M3 is not supported");
        case DataType::INVALID: TT_THROW("StreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Chunk plan: largest pages_per_chunk that fits the scratch CB and divides
// tensor_num_pages evenly.
struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
};

inline ChunkPlan derive_chunk_plan(
    uint32_t tensor_page_size, uint32_t tensor_num_pages, uint32_t scratch_cb_size_bytes) {
    TT_FATAL(tensor_page_size > 0, "device_tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "device_tensor must have at least one page");
    TT_FATAL(
        scratch_cb_size_bytes >= tensor_page_size,
        "scratch_cb_size_bytes ({} B) must be >= tensor page size ({} B); "
        "consider a layout with smaller pages or a larger CB budget",
        scratch_cb_size_bytes,
        tensor_page_size);

    const uint32_t max_pages_per_chunk_by_cb = scratch_cb_size_bytes / tensor_page_size;
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk_by_cb);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    return ChunkPlan{
        .socket_page_size = pages_per_chunk * tensor_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
    };
}

// ---------------------------------------------------------------------------
// D2H split-kernel chunk plan (reader/writer pipeline).
//
// Like the shared ChunkPlan but adds slot_count: the data-CB depth (in full
// socket-page slots) that lets the reader stage DRAM pages ahead while the
// writer drains earlier ones into the host-pinned socket FIFO. Mirrors the
// H2DChunkPlan approach: size the socket page to a few NOC bursts (capped by
// max_socket_page_size_bytes), then fill the remaining service-core L1 with
// slots above a double-buffering floor.
// ---------------------------------------------------------------------------

// Host-side sizing heuristics (same as H2D — the reader chunks each socket
// page by the real device NOC burst internally; kNocBurstBytes is only a host
// granularity target, NOT device truth).
inline constexpr uint32_t kD2HNocBurstBytes = 16u * 1024;
inline constexpr uint32_t kD2HTargetReadBursts = 8;  // default socket-page target ~= 8 bursts (128 KB)
inline constexpr uint32_t kD2HSlotCap = 64;          // upper bound on data-CB slots
inline constexpr uint32_t kD2HMinDataSlots = 2;      // double-buffering floor (reader/writer overlap)

// The data CB (program allocator, bottom-up) and the service-core scratch
// (ServiceCoreManager, top-down) share the unreserved L1 with no cross-allocator
// overflow check. This reserve holds back headroom for the socket config buffer
// and post-plan scratch words (termination, worker-sync counters, metadata
// staging) plus a safety pad. Generous on purpose (>> their sum).
inline constexpr uint64_t kD2HServiceScratchReserveBytes = 16u * 1024;

struct D2HChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
    uint32_t slot_count;        // full-page data-CB slots backing the reader/writer pipeline
};

inline D2HChunkPlan derive_d2h_chunk_plan(
    uint32_t tensor_page_size, uint32_t tensor_num_pages, uint64_t usable_cb_l1_bytes, uint64_t page_budget_hint) {
    TT_FATAL(tensor_page_size > 0, "D2HStreamService: tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "D2HStreamService: tensor must have at least one page");
    TT_FATAL(
        tensor_page_size <= usable_cb_l1_bytes,
        "D2HStreamService: tensor page {} B exceeds service-core CB L1 budget {} B; "
        "use a layout with smaller pages",
        tensor_page_size,
        usable_cb_l1_bytes);
    TT_FATAL(
        page_budget_hint == 0 || page_budget_hint >= tensor_page_size,
        "D2HStreamService: max_socket_page_size_bytes ({} B) must be >= the tensor page size ({} B); "
        "the socket page can't be smaller than one tensor page (pass 0 to auto-size)",
        page_budget_hint,
        tensor_page_size);

    const uint64_t burst_target = static_cast<uint64_t>(kD2HTargetReadBursts) * kD2HNocBurstBytes;
    const uint64_t requested = page_budget_hint > 0 ? page_budget_hint : burst_target;
    const uint64_t page_budget = std::min<uint64_t>(requested, usable_cb_l1_bytes / kD2HMinDataSlots);

    uint32_t pages_per_chunk = std::max<uint32_t>(1, static_cast<uint32_t>(page_budget / tensor_page_size));
    pages_per_chunk = std::min(pages_per_chunk, tensor_num_pages);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    const uint32_t socket_page_size = pages_per_chunk * tensor_page_size;

    const uint32_t slots_l1_max = static_cast<uint32_t>(usable_cb_l1_bytes / socket_page_size);
    const uint32_t slot_count = std::min(slots_l1_max, kD2HSlotCap);
    if (slot_count < kD2HMinDataSlots) {
        log_warning(
            tt::LogOp,
            "D2HStreamService: tensor page {} B leaves only {} CB slot(s) in {} B of L1; "
            "reader/writer overlap disabled (use a smaller per-shard page to double-buffer)",
            tensor_page_size,
            slot_count,
            usable_cb_l1_bytes);
    }
    return D2HChunkPlan{
        .socket_page_size = socket_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
        .slot_count = slot_count,
    };
}

}  // namespace tt::tt_metal
