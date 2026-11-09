// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_new {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorSpec;

// PageCopy moves at most this many bytes of pages per barrier.
constexpr uint32_t kPageCopyBatchBytes = 16 * 1024;
constexpr uint32_t kPageCopyMaxBatch = 16;
// ROW_MAJOR output pages wider than this are split into chunks of this size (a multiple of 16 bytes).
constexpr uint32_t kRowChunkBytes = 4 * 1024;
// Per stage entry: fragment records and staged fragment bytes. A unit needing more spills into more entries.
constexpr uint32_t kRecordsPerEntry = 128;
constexpr uint32_t kStageDataBytes = 8 * 1024;
constexpr uint32_t kNumStageEntries = 3;
// L1 addresses written by the NOC from a byte-granular source must share the source's offset within 16 bytes.
constexpr uint32_t kNocWordBytes = 16;

uint32_t dim_from_end(const tt::tt_metal::Shape& shape, uint32_t from_end) {
    return shape.rank() >= from_end ? shape[-static_cast<int>(from_end)] : 1;
}

// Bytes per element for the dtypes the Gather kernels move element-wise. Packed formats have none.
uint32_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::UINT16: return 2;
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32: return 4;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::FP8_E4M3: return 1;
        default: return 0;
    }
}

// A config normalized from an ND shard spec keeps that spec next to its 2D equivalent, and a view of the tensor
// drops it (view_device), so two such configs place pages alike when their 2D specs agree.
bool same_distribution(const tt::tt_metal::MemoryConfig& a, const tt::tt_metal::MemoryConfig& b) {
    if (is_functionally_same_memory_config(a, b)) {
        return true;
    }
    return a.shard_spec().has_value() && b.shard_spec().has_value() &&
           (a.created_with_nd_shard_spec() || b.created_with_nd_shard_spec()) &&
           is_functionally_same_memory_config(drop_normalized_nd_shard_spec(a), drop_normalized_nd_shard_spec(b));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ReshapeSideGeometry side_geometry(const tt::tt_metal::TensorSpec& spec) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& logical = spec.logical_shape();
    const auto& padded = spec.padded_shape();
    const uint32_t rank = std::max(logical.rank(), padded.rank());

    uint64_t batch = 1;
    for (uint32_t from_end = 3; from_end <= rank; ++from_end) {
        const uint32_t logical_dim = dim_from_end(logical, from_end);
        TT_FATAL(
            logical_dim == dim_from_end(padded, from_end),
            "reshape: padding is only supported in the last two dims, but dim -{} of {} is padded ({} -> {}). "
            "Tensor spec: {}",
            from_end,
            logical,
            logical_dim,
            dim_from_end(padded, from_end),
            spec);
        batch *= logical_dim;
    }

    const auto& physical = spec.physical_shape();
    TT_FATAL(
        static_cast<uint64_t>(physical.height()) * physical.width() <= std::numeric_limits<uint32_t>::max(),
        "reshape: tensors of more than 2^32 physical elements are not supported. Tensor spec: {}",
        spec);

    ReshapeSideGeometry geometry;
    geometry.width = dim_from_end(logical, 1);
    geometry.height = dim_from_end(logical, 2);
    geometry.padded_height = dim_from_end(padded, 2);
    geometry.batch = static_cast<uint32_t>(batch);
    if (spec.layout() == Layout::TILE) {
        const auto tile = spec.tile().get_tile_shape();
        geometry.page_width = tile[1];
        geometry.pages_per_row = physical.width() / tile[1];
        geometry.num_pages = (physical.height() / tile[0]) * geometry.pages_per_row;
    } else {
        const uint32_t element_size = element_size_bytes(spec.data_type());
        const auto page_bytes = static_cast<uint32_t>(spec.compute_page_size_bytes());
        geometry.page_width = element_size != 0 ? page_bytes / element_size : physical.width();
        geometry.pages_per_row = geometry.page_width != 0 ? physical.width() / geometry.page_width : 0;
        geometry.num_pages = physical.height() * geometry.pages_per_row;
    }
    return geometry;
}

bool page_sequences_identical(const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output) {
    if (input.data_type() != output.data_type() || input.layout() != output.layout() || input.tile() != output.tile() ||
        input.physical_shape() != output.physical_shape() ||
        input.compute_page_size_bytes() != output.compute_page_size_bytes()) {
        return false;
    }
    const auto& in_padded = input.padded_shape();
    const auto& out_padded = output.padded_shape();
    if (CMAKE_UNIQUE_NAMESPACE::dim_from_end(in_padded, 1) != CMAKE_UNIQUE_NAMESPACE::dim_from_end(out_padded, 1)) {
        return false;
    }
    const auto in = side_geometry(input);
    const auto out = side_geometry(output);
    if (in.width != out.width || in.page_width != out.page_width) {
        return false;
    }
    const bool same_rows = in.height == out.height && in.padded_height == out.padded_height;
    const bool no_row_padding = in.height == in.padded_height && out.height == out.padded_height;
    return same_rows || no_row_padding;
}

ReshapeKind plan_reshape(const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output) {
    const auto& in_config = input.memory_config();
    const auto& out_config = output.memory_config();
    const bool same_config = CMAKE_UNIQUE_NAMESPACE::same_distribution(in_config, out_config);

    if (input.logical_shape() == output.logical_shape() && input.padded_shape() == output.padded_shape() &&
        input.data_type() == output.data_type() && input.layout() == output.layout() && same_config) {
        return ReshapeKind::NoOp;
    }
    if (!page_sequences_identical(input, output)) {
        return ReshapeKind::Gather;
    }
    // An ND shard spec is written against the tensor's own shape, so a view can keep it only when the padded
    // shape (rank included) does not change.
    const bool nd_sharded = is_nd_sharded_memory_config(in_config) || is_nd_sharded_memory_config(out_config);
    if (same_config && (!nd_sharded || input.padded_shape() == output.padded_shape())) {
        return ReshapeKind::View;
    }
    return ReshapeKind::PageCopy;
}

PageCopyParams make_page_copy_params(
    const tt::tt_metal::TensorSpec& input,
    const tt::tt_metal::TensorSpec& output,
    uint32_t input_alignment,
    uint32_t output_alignment) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const uint32_t alignment = std::max({input_alignment, output_alignment, kNocWordBytes});
    PageCopyParams params;
    params.num_pages = side_geometry(output).num_pages;
    params.page_bytes = static_cast<uint32_t>(output.compute_page_size_bytes());
    TT_FATAL(
        input.compute_page_size_bytes() == params.page_bytes,
        "reshape: a page copy needs equal page sizes. Input spec: {}, output spec: {}",
        input,
        output);
    if (params.page_bytes <= kPageCopyBatchBytes) {
        params.chunk_bytes = params.page_bytes;
        params.chunks_per_page = 1;
    } else {
        params.chunk_bytes = tt::round_down(kPageCopyBatchBytes, alignment);
        params.chunks_per_page = tt::div_up(params.page_bytes, params.chunk_bytes);
    }
    params.num_units = params.num_pages * params.chunks_per_page;
    params.entry_bytes = tt::round_up(params.chunk_bytes, alignment);
    params.units_per_batch = std::clamp<uint32_t>(kPageCopyBatchBytes / params.entry_bytes, 1, kPageCopyMaxBatch);
    params.num_entries = 2 * params.units_per_batch;
    return params;
}

GatherParams make_gather_params(
    const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output, uint32_t input_alignment) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GatherParams params;
    params.in = side_geometry(input);
    params.out = side_geometry(output);
    params.tiled = input.layout() == Layout::TILE;
    params.element_size = element_size_bytes(input.data_type());
    TT_FATAL(
        params.element_size != 0,
        "reshape: an element-wise gather does not support dtype {}. Input spec: {}, output spec: {}",
        input.data_type(),
        input,
        output);
    TT_FATAL(
        input.data_type() == output.data_type() && input.layout() == output.layout(),
        "reshape: input and output must share dtype and layout. Input spec: {}, output spec: {}",
        input,
        output);

    params.page_bytes = static_cast<uint32_t>(output.compute_page_size_bytes());
    uint32_t max_fragment_bytes = 0;
    if (params.tiled) {
        const auto& tile = input.tile();
        TT_FATAL(
            tile == output.tile() && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
            "reshape: transposed tiles are not supported. Input spec: {}, output spec: {}",
            input,
            output);
        params.tile_height = tile.get_tile_shape()[0];
        params.tile_width = tile.get_tile_shape()[1];
        params.face_height = tile.get_face_shape()[0];
        params.face_width = tile.get_face_shape()[1];
        params.unit_bytes = params.page_bytes;
        params.chunks_per_page = 1;
        max_fragment_bytes = params.face_width * params.element_size;
    } else {
        if (params.page_bytes <= kRowChunkBytes) {
            params.unit_bytes = params.page_bytes;
            params.chunks_per_page = 1;
        } else {
            params.unit_bytes = kRowChunkBytes;
            params.chunks_per_page = tt::div_up(params.page_bytes, kRowChunkBytes);
        }
        max_fragment_bytes = std::min(params.unit_bytes, params.in.page_width * params.element_size);
    }
    params.num_units = params.out.num_pages * params.chunks_per_page;

    params.source_alignment = std::max(input_alignment, kNocWordBytes);
    params.records_per_entry = kRecordsPerEntry;
    params.header_bytes =
        tt::round_up((kGatherHeaderWords + kGatherRecordWords * kRecordsPerEntry) * sizeof(uint32_t), kNocWordBytes);
    // Worst case for one fragment: its bytes plus the misalignment of its source, rounded to a NOC word.
    const uint32_t worst_fragment_span =
        tt::round_up(max_fragment_bytes + params.source_alignment + kNocWordBytes, kNocWordBytes);
    params.stage_data_bytes = std::max(kStageDataBytes, worst_fragment_span);
    params.stage_entry_bytes = params.header_bytes + params.stage_data_bytes;
    params.num_stage_entries = kNumStageEntries;
    return params;
}

}  // namespace ttnn::operations::data_movement::reshape_new
