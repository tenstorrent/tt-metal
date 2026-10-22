// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Output CB depth is two batches, so reader and writer can each own one.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch) { return 2 * std::max(read_batch, write_batch); }

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;

// Batching the CB plan starts from; plan_batches halves it until the plan fits the L1 it is given.
constexpr uint32_t kBatch = 4;

bool dtype_ok(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::INT32:
        case DataType::UINT32: return true;
        default: return false;
    }
}

ttnn::Shape concat_output_shape(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    ttnn::Shape shape = input_tensors.front().logical_shape();
    shape[dim] = 0;
    for (const Tensor& input : input_tensors) {
        shape[dim] += input.logical_shape()[dim];
    }
    return shape;
}

bool interleaved_in_dram_or_l1(const tt::tt_metal::MemoryConfig& config) {
    return config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
           (config.buffer_type() == BufferType::DRAM || config.buffer_type() == BufferType::L1);
}

bool inputs_supported(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    if (input_tensors.size() < 2 || input_tensors.size() > kConcatMaxNwayInputs) {
        return false;
    }
    const Tensor& first = input_tensors.front();
    if (first.storage_type() != ttnn::StorageType::DEVICE || first.buffer() == nullptr) {
        return false;
    }
    const uint32_t rank = first.logical_shape().rank();
    if (rank == 0 || attrs.dim >= rank) {
        return false;
    }
    for (const Tensor& input : input_tensors) {
        if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr ||
            input.device() != first.device()) {
            return false;
        }
        if (input.layout() != ttnn::ROW_MAJOR_LAYOUT || input.dtype() != first.dtype() || !dtype_ok(input.dtype())) {
            return false;
        }
        if (!interleaved_in_dram_or_l1(input.memory_config())) {
            return false;
        }
        // The kernels derive sticks from the logical shape, and an empty input leaves a core no work.
        if (input.padded_shape() != input.logical_shape() || input.logical_shape().volume() == 0 ||
            input.logical_shape().rank() != rank) {
            return false;
        }
        for (uint32_t d = 0; d < rank; ++d) {
            if (d != attrs.dim && input.logical_shape()[d] != first.logical_shape()[d]) {
                return false;
            }
        }
    }
    return true;
}

uint64_t l1_output_reservation(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    return get_pending_l1_output_reservation(
        first, concat_output_shape(input_tensors, attrs.dim), attrs.output_mem_config, first.dtype(), first.layout());
}

uint64_t saturating_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

}  // namespace

bool concats_width(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    return attrs.dim + 1 == input_tensors.front().logical_shape().rank();
}

bool is_two_input_leaf(const std::vector<Tensor>& input_tensors) { return input_tensors.size() == 2; }

CbPages cb_pages(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    const auto out_shape = concat_output_shape(input_tensors, attrs.dim);
    const uint32_t out_stick = out_shape[kLastAxis] * first.element_size();

    // A slot offset inside the CB is a NoC destination for the reads, so it must be legal for every source's transport;
    // the writer's page pitch in the output buffer follows the output's transport alignment.
    uint32_t cb_alignment = tt::tt_metal::hal::get_l1_alignment();
    if (attrs.output_mem_config.buffer_type() == BufferType::DRAM) {
        cb_alignment = std::max(cb_alignment, tt::tt_metal::hal::get_dram_alignment());
    }
    uint32_t max_input_page = 0;
    for (const Tensor& input : input_tensors) {
        if (input.memory_config().buffer_type() == BufferType::DRAM) {
            cb_alignment = std::max(cb_alignment, tt::tt_metal::hal::get_dram_alignment());
        }
        max_input_page = std::max(max_input_page, static_cast<uint32_t>(input.buffer()->aligned_page_size()));
    }

    if (concats_width(input_tensors, attrs)) {
        // The two-input reader stages through a scratch page with one L1 alignment of slack.
        const uint32_t scratch_slack = is_two_input_leaf(input_tensors) ? tt::tt_metal::hal::get_l1_alignment() : 0;
        return CbPages{
            .out_page = tt::round_up(out_stick, cb_alignment), .scratch_page = max_input_page + scratch_slack};
    }
    return CbPages{.out_page = tt::round_up(std::max(out_stick, max_input_page), cb_alignment), .scratch_page = 0};
}

namespace {

uint64_t footprint_bytes(const CbPages& pages, const BatchPlan& batches) {
    const uint64_t out_bytes =
        static_cast<uint64_t>(cb_depth(batches.read_batch, batches.write_batch)) * pages.out_page;
    if (pages.scratch_page == 0) {
        return out_bytes;
    }
    // Every CB starts on the allocator's alignment, so the scratch CB sits behind the padding of the output CB.
    const uint64_t cb_start_alignment =
        std::max(tt::tt_metal::hal::get_l1_alignment(), tt::tt_metal::hal::get_dram_alignment());
    return tt::round_up(out_bytes, cb_start_alignment) + pages.scratch_page;
}

}  // namespace

uint64_t cb_footprint_bytes(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, const BatchPlan& batches) {
    return footprint_bytes(cb_pages(input_tensors, attrs), batches);
}

// The plan runs inside the per-call program hash, so the page geometry is derived once, not per halving step.
std::optional<BatchPlan> plan_batches(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, uint64_t budget) {
    const CbPages pages = cb_pages(input_tensors, attrs);
    BatchPlan batches{kBatch, kBatch};
    while (footprint_bytes(pages, batches) > budget) {
        if (batches.read_batch == 1 && batches.write_batch == 1) {
            return std::nullopt;
        }
        batches.read_batch = std::max(1u, batches.read_batch / 2);
        batches.write_batch = std::max(1u, batches.write_batch / 2);
    }
    return batches;
}

std::optional<BatchPlan> plan_batches_live(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, bool output_allocated) {
    // An allocated output has already lowered the live frontier; charging it again would shrink the plan.
    const uint64_t live = get_max_l1_space(input_tensors.front());
    const uint64_t budget = output_allocated ? live : saturating_sub(live, l1_output_reservation(input_tensors, attrs));
    return plan_batches(input_tensors, attrs, budget);
}

bool fits_live_l1(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    return plan_batches_live(input_tensors, attrs, /*output_allocated=*/false).has_value();
}

tt::tt_metal::TensorSpec codegen_output_spec(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    return tt::tt_metal::TensorSpec(
        concat_output_shape(input_tensors, attrs.dim),
        tt::tt_metal::TensorLayout(
            first.dtype(),
            tt::tt_metal::PageConfig(first.layout(), first.tensor_spec().tile()),
            attrs.output_mem_config));
}

bool supported_by_codegen(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    if (!inputs_supported(input_tensors, attrs) || !interleaved_in_dram_or_l1(attrs.output_mem_config)) {
        return false;
    }
    // Static window: the output allocation between routing and validate would flip a live answer.
    const uint64_t budget =
        saturating_sub(get_static_l1_space(input_tensors.front()), l1_output_reservation(input_tensors, attrs));
    return plan_batches(input_tensors, attrs, budget).has_value();
}

namespace {

using Shape = std::vector<uint32_t>;
using Placement = std::vector<BufferType>;

bool shape_matches(const Tensor& tensor, const Shape& expected) {
    const auto& logical = tensor.logical_shape();
    if (logical.rank() != expected.size()) {
        return false;
    }
    for (size_t d = 0; d < expected.size(); ++d) {
        if (logical[d] != expected[d]) {
            return false;
        }
    }
    return true;
}

bool inputs_match(
    const std::vector<Tensor>& input_tensors, const std::vector<Shape>& shapes, const Placement& placement) {
    if (input_tensors.size() != shapes.size() || input_tensors.size() != placement.size()) {
        return false;
    }
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        if (!shape_matches(input_tensors[i], shapes[i]) ||
            input_tensors[i].memory_config().buffer_type() != placement[i]) {
            return false;
        }
    }
    return true;
}

bool dtype_in(DataType dtype, std::initializer_list<DataType> dtypes) {
    return std::find(dtypes.begin(), dtypes.end(), dtype) != dtypes.end();
}

// Exact match on the normalized attributes of one measured last-axis case: dtype, per-input shape and buffer
// placement, and an interleaved DRAM output.
bool is_exact_case(
    const std::vector<Tensor>& input_tensors,
    const ttnn::prim::ConcatCodegenParams& attrs,
    std::initializer_list<DataType> dtypes,
    const std::vector<Shape>& shapes,
    const Placement& placement) {
    const auto last_dim = static_cast<uint32_t>(shapes.front().size() - 1);
    return attrs.dim == last_dim && attrs.output_mem_config.buffer_type() == BufferType::DRAM &&
           inputs_match(input_tensors, shapes, placement) && dtype_in(input_tensors.front().dtype(), dtypes);
}

}  // namespace

bool is_demoted(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    constexpr auto kDram = BufferType::DRAM;
    constexpr auto kL1 = BufferType::L1;
    constexpr auto kBf16 = DataType::BFLOAT16;
    constexpr auto kInt32 = DataType::INT32;
    constexpr auto kUint32 = DataType::UINT32;

    // Every measured extent is a multiple of one tile edge.
    constexpr uint32_t kUnit = 32;
    constexpr uint32_t kDouble = 2 * kUnit;
    constexpr uint32_t kTriple = kUnit + kDouble;

    // Unsettled: measured slower than the native path, mechanism not analyzed; matched case by case.
    const std::vector<Shape> three_input_rank3 = {{1, kUnit, kUnit}, {1, kUnit, kDouble}, {1, kUnit, kUnit}};
    const std::vector<Placement> placements = {{kDram, kL1, kDram}, {kL1, kDram, kL1}, {kDram, kDram, kDram}};
    for (const Placement& placement : placements) {
        if (is_exact_case(input_tensors, attrs, {kBf16, kInt32, kUint32}, three_input_rank3, placement)) {
            return true;
        }
    }

    // Unsettled: same as above, width concat of three rank-2 inputs.
    const std::vector<Shape> three_input_width = {{kUnit, kUnit}, {kUnit, kDouble}, {kUnit, kTriple}};
    for (const Placement& placement : placements) {
        if (is_exact_case(input_tensors, attrs, {kBf16, kInt32, kUint32}, three_input_width, placement)) {
            return true;
        }
    }
    return false;
}

bool supported_execution_controls(
    unsigned int groups, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return groups == 1 && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::concat_codegen
