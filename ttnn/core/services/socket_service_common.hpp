// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal helpers shared by h2d_socket_service.cpp and d2h_socket_service.cpp.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/data_types.hpp>

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

}  // namespace tt::tt_metal
