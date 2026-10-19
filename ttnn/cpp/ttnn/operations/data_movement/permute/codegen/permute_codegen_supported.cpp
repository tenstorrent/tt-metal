// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_supported.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

namespace {

constexpr uint32_t kMinRank = 2;
constexpr uint32_t kMaxRank = ttnn::prim::kPermuteCodegenMaxDims;
constexpr uint32_t kWidthAxisFromEnd = ttnn::prim::kPermuteWidthAxisFromEnd;
constexpr uint32_t kHeightAxisFromEnd = ttnn::prim::kPermuteHeightAxisFromEnd;

// The fused height/width transpose route is the better transport for deep stacks of
// whole-tile matrices; the working set is one tilized matrix held in L1.
constexpr uint32_t kFusedMinMatrices = 6;
constexpr uint64_t kFusedMaxMatrixBytes = 1024 * 1024;
constexpr uint32_t kNarrowElementBytes = 2;
constexpr uint32_t kWideElementBytes = 4;

bool dtype_supported(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::FLOAT32 ||
           dtype == tt::tt_metal::DataType::INT32;
}

bool is_fused_height_width_case(
    const ttnn::Shape& shape, const ttsl::SmallVector<uint32_t>& dims, uint32_t element_size) {
    const uint32_t rank = shape.rank();
    if (dims[rank - kWidthAxisFromEnd] != rank - kHeightAxisFromEnd) {
        return false;
    }
    const uint64_t height = shape[rank - kHeightAxisFromEnd];
    const uint64_t width = shape[rank - kWidthAxisFromEnd];
    const uint64_t matrices = shape.volume() / (height * width);
    const bool element_ok = element_size == kNarrowElementBytes || element_size == kWideElementBytes;
    const bool whole_tiles = height % tt::constants::TILE_HEIGHT == 0 && width % tt::constants::TILE_WIDTH == 0;
    const uint64_t matrix_bytes = height * width * element_size;
    return element_ok && whole_tiles && matrices >= kFusedMinMatrices && matrix_bytes <= kFusedMaxMatrixBytes;
}

bool geometry_fits(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& dims, const tt::tt_metal::MemoryConfig& output_mem_config) {
    const ttnn::Shape& shape = input.logical_shape();
    const uint32_t rank = shape.rank();
    const uint32_t element_size = input.element_size();
    const uint32_t last_output_axis = dims[rank - kWidthAxisFromEnd];

    const auto& allocator = input.device()->allocator();
    const uint64_t out_alignment = allocator->get_alignment(output_mem_config.buffer_type());
    const tt::tt_metal::TensorSpec output_spec = permute_output_spec(input, dims, output_mem_config);
    const uint64_t in_page = input.buffer()->aligned_page_size();
    const uint64_t out_page = tt::round_up(uint64_t{output_spec.compute_page_size_bytes()}, out_alignment);

    const uint64_t l1_budget = ttnn::prim::permute_cb_l1_budget(input, output_spec);
    if (last_output_axis == rank - kWidthAxisFromEnd) {
        const auto slot_bytes = static_cast<uint32_t>(std::max(in_page, out_page));
        return ttnn::prim::plan_permute_rm_cb(slot_bytes, l1_budget).has_value();
    }
    return ttnn::prim::permute_blocked_cb_bytes(element_size) <= l1_budget;
}

}  // namespace

bool is_valid_permutation(const ttsl::SmallVector<uint32_t>& dims, uint32_t rank) {
    std::vector<uint32_t> identity(rank);
    std::iota(identity.begin(), identity.end(), 0);
    return dims.size() == rank && std::is_permutation(dims.cbegin(), dims.cend(), identity.cbegin());
}

bool supported_by_codegen(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& dims, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return false;
    }
    if (input.layout() != ttnn::ROW_MAJOR_LAYOUT || !dtype_supported(input.dtype())) {
        return false;
    }
    // Neither end of the copy is addressed through a shard map.
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    // The hash plans the CB before the output exists, the factory after; an L1 output would move the live frontier
    // between them.
    if (output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1) {
        return false;
    }
    const ttnn::Shape& shape = input.logical_shape();
    const uint32_t rank = shape.rank();
    if (rank < kMinRank || rank > kMaxRank || !is_valid_permutation(dims, rank) || shape.volume() == 0) {
        return false;
    }
    if (is_fused_height_width_case(shape, dims, input.element_size())) {
        return false;
    }
    return geometry_fits(input, dims, output_mem_config);
}

bool is_demoted(const Tensor& /*input*/, const ttsl::SmallVector<uint32_t>& /*dims*/) {
    // No shape is perf-demoted. The gate stays as the routing extension point for a
    // genuine device regression on a shape the correctness gate admits.
    return false;
}

tt::tt_metal::TensorSpec permute_output_spec(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& dims, const tt::tt_metal::MemoryConfig& output_mem_config) {
    const ttnn::Shape& input_shape = input.logical_shape();
    ttsl::SmallVector<uint32_t> output_shape;
    output_shape.reserve(dims.size());
    std::transform(dims.cbegin(), dims.cend(), std::back_inserter(output_shape), [&input_shape](uint32_t d) {
        return input_shape[d];
    });
    return tt::tt_metal::TensorSpec(
        ttnn::Shape(std::move(output_shape)),
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()), output_mem_config));
}

bool output_spec_matches(const tt::tt_metal::TensorSpec& expected, const tt::tt_metal::TensorSpec& actual) {
    return expected.logical_shape() == actual.logical_shape() && expected.data_type() == actual.data_type() &&
           expected.page_config() == actual.page_config() && expected.memory_config() == actual.memory_config();
}

}  // namespace ttnn::operations::data_movement::permute_codegen
